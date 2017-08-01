import datetime as dt
import logging
from pathlib import Path
import os


from bokeh import events
from bokeh.colors import RGB
from bokeh.layouts import layout
from bokeh.models import (
    Range1d, LinearColorMapper, ColorBar, FixedTicker,
    ColumnDataSource, CustomJS, WMTSTileSource)
from bokeh.plotting import figure, curdoc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable, get_cmap
import numpy as np
from tornado import gen


MIN_VAL = 0.01
MAX_VAL = 2
ALPHA = 0.7
DATA_DIRECTORY = os.getenv('MRMS_DATADIR', '~/.mrms')


def load_data(date='latest'):
    strformat = '%Y/%m/%d/%HZ.npz'
    dir = os.path.expanduser(DATA_DIRECTORY)
    if date == 'latest':
        p = Path(dir)
        path = sorted([pp for pp in p.rglob('*.npz')], reverse=True)[0]
    else:
        path = os.path.join(dir, date.strftime(strformat))

    valid_date = dt.datetime.strptime(str(path), '{}/{}'.format(dir, strformat))
    data_load = np.load(path)
    regridded_data = data_load['data'] / 25.4  # mm to in
    X = data_load['X']
    Y = data_load['Y']
    masked_regrid = np.ma.masked_less(regridded_data, MIN_VAL).clip(
        max=MAX_VAL)
    return masked_regrid, X, Y, valid_date


def find_all_times():
    p = Path(DATA_DIRECTORY).expanduser()
    out = {}
    for pp in p.rglob('*.npz'):
        try:
            datetime = dt.datetime.strptime(''.join(pp.parts[-4:]),
                                            '%Y%m%d%HZ.npz')
        except ValueError:
            logging.debug('%s does not conform to expected format', pp)
            continue
        date = datetime.strftime('%Y-%m-%d')
        if date not in out:
            out[date] = []
        out[date].append(datetime.strftime('%HZ'))
    return out


# setup the coloring
levels = MaxNLocator(nbins=21).tick_values(0, MAX_VAL)
cmap = get_cmap('viridis')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
sm = ScalarMappable(norm=norm, cmap=cmap)
color_pal = [RGB(*val).to_hex() for val in
             sm.to_rgba(levels, bytes=True, norm=True)[:-1]]
color_mapper = LinearColorMapper(color_pal, low=sm.get_clim()[0],
                                 high=sm.get_clim()[1])
ticker = FixedTicker(ticks=levels[::3])
cb = ColorBar(color_mapper=color_mapper, location=(0, 0),
              scale_alpha=ALPHA, ticker=ticker)

masked_regrid, X, Y, valid_date = load_data()

xn = X[0]
yn = Y[:, 0]
dw = xn[-1] - xn[0]
dh = yn[-1] - yn[0]
dx = xn[1] - xn[0]
dy = yn[1] - yn[0]

# make histograms
counts, bin_edges = np.histogram(masked_regrid, bins=levels,
                                 range=(levels.min(), levels.max()))
bin_width = bin_edges[1] - bin_edges[0]
bin_centers = bin_edges + bin_width / 2
hist_sources = [ColumnDataSource(data={'x': [bin_centers[i]],
                                       'top': [counts[i]],
                                       'color': [color_pal[i]]})
                for i in range(len(counts))]

# bokeh plotting
rgba_vals = sm.to_rgba(masked_regrid, bytes=True, alpha=ALPHA)

width = 600
height = int(width / 2 * dy / dx)
sfmt = '%Y-%m-%d %HZ'
title = 'MRMS Precipitation {} through {}'.format(
    (valid_date - dt.timedelta(hours=24)).strftime(sfmt),
    valid_date.strftime(sfmt))
map_fig = figure(plot_width=width, plot_height=height,
                 x_range=(xn[0], xn[-1]), y_range=(yn[0], yn[-1]),
                 y_axis_type=None, x_axis_type=None,
                 toolbar_location='left',
                 title=title)

rgba_img = map_fig.image_rgba([rgba_vals], x=[xn[0]], y=[yn[0]], dw=[dw],
                              dh=[dh])
# Need to use this and not bokeh.tile_providers.STAMEN_TONER
# https://github.com/bokeh/bokeh/issues/4770
STAMEN_TONER = WMTSTileSource(
    url='https://stamen-tiles.a.ssl.fastly.net/toner-lite/{Z}/{X}/{Y}.png',
    attribution=(
        'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
        'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '
        'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
        'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
    )
)
map_fig.add_tile(STAMEN_TONER)
map_fig.add_layout(cb, 'right')

# Make the histogram figure
hist_fig = figure(plot_width=height, plot_height=height,
                  toolbar_location='right',
                  x_axis_label='Precipitation (inches)',
                  y_axis_label='Counts',
                  x_range=Range1d(start=0, end=MAX_VAL))

for source in hist_sources:
    hist_fig.vbar(x='x', top='top', width=bin_width, bottom=0, color='color',
                  fill_alpha=ALPHA, source=source)

line_source = ColumnDataSource(data={'x': [-1, -1], 'y': [0, counts.max()]})
hist_fig.line(x='x', y='y', color='red', source=line_source, alpha=ALPHA)


# Dynamic histogram update and bin indicator line
pos_source = ColumnDataSource(data={
    'bin_centers': [bin_centers], 'shape': [masked_regrid.shape[1]],
    'bin': [np.digitize(masked_regrid, levels[:-1]).astype('uint8').ravel()],
    'dx': [dx], 'dy': [dy], 'left': [xn[0]], 'bottom': [yn[0]]})

line_callback = CustomJS(args={'source': pos_source,
                               'lsource': line_source},
                         code="""
var data = source.data;
var x = cb_obj['x'];
var y = cb_obj['y'];

var x_index = 0;
var y_index = 0;

x_index = Math.round((x - data['left'][0])/data['dx'][0]);
y_index = Math.round((y - data['bottom'][0])/data['dy'][0]);
var idx = y_index * data['shape'][0] + x_index;
var bin = data['bin'][0][idx];
var center = data['bin_centers'][0][bin - 1];
var ldata = lsource.data;
xi = ldata['x'];
xi[0] = center;
xi[1] = center;
setTimeout(function(){lsource.change.emit()}, 100);
""")

no_line = CustomJS(args={'lsource': line_source}, code="""
var data = lsource.data;
xi = data['x'];
xi[0] = -1;
xi[1] = -1;
lsource.change.emit();
""")


def update_histogram(attr, old, new):
    # makes it so only one callback added per 100 ms
    try:
        doc.add_timeout_callback(_update_histogram, 100)
    except ValueError:
        pass

@gen.coroutine
def _update_histogram():
    left = map_fig.x_range.start
    right = map_fig.x_range.end
    bottom = map_fig.y_range.start
    top = map_fig.y_range.end
    left_idx = np.abs(xn - left).argmin()
    right_idx = np.abs(xn - right).argmin() + 1
    bottom_idx = np.abs(yn - bottom).argmin()
    top_idx = np.abs(yn - top).argmin() + 1
    logging.debug('Updating histogram...')

    counts, _ = np.histogram(
        masked_regrid[bottom_idx:top_idx, left_idx:right_idx], bins=levels,
        range=(levels.min(), levels.max()))
    line_source.data.update({'y': [0, counts.max()]})
    for i, source in enumerate(hist_sources):
        source.data.update({'top': [counts[i]]})


map_fig.js_on_event(events.MouseMove, line_callback)
map_fig.js_on_event(events.MouseLeave, no_line)
map_fig.x_range.on_change('start', update_histogram)
map_fig.x_range.on_change('end', update_histogram)
map_fig.y_range.on_change('start', update_histogram)
map_fig.y_range.on_change('end', update_histogram)

# layout the document
l = layout([[map_fig, hist_fig]])
doc = curdoc()
doc.add_root(l)
