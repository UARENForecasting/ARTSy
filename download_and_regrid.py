#! /usr/bin/env python
"""
Download the MRMS data, get only the given area, project to web mercator,
and regrid onto a regular grid for later plotting.
"""
import argparse
import datetime as dt
import gzip
from io import BytesIO
import logging
import os
import shutil
import sys
import tempfile


import numpy as np
import pygrib
import requests
import scipy.interpolate


def webmerc_proj(lat, lon):
    """Convert latititude and longitude to web mercator"""
    R = 6378137
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def download_data(date, tmpfile):
    """Download the MRMS correlated precip data from NCEP"""
    base_url = 'http://mrms.ncep.noaa.gov/data/2D/GaugeCorr_QPE_24H/'
    if date == 'latest':
        timestr = '.latest'
    else:
        timestr = date.strftime('_00.00_%Y%m%d-%H%M%S')
    filename = 'MRMS_GaugeCorr_QPE_24H{timestr}.grib2.gz'.format(
            timestr=timestr)
    logging.info('Making request for %s', filename)
    r = requests.get(base_url + filename)
    if r.status_code != 200:
        logging.error('Failed to retrieve file: %s', r.text)
        sys.exit(1)
    gzipped_data = BytesIO(r.content)
    with gzip.open(gzipped_data, 'rb') as f_in:
        shutil.copyfileobj(f_in, tmpfile)
    tmpfile.flush()


def read_subset(tmpfilename, bbox):
    """Read a subset of the data from the grib file"""
    logging.info('Reading subset of data from grib file')
    grbs = pygrib.open(tmpfilename)
    grb = grbs.message(1)
    min_lat, max_lat, min_lon, max_lon = bbox
    grb_data, lats, lons = grb.data(lat1=min_lat, lat2=max_lat, lon1=min_lon,
                                    lon2=max_lon)
    valid_date = grb.validDate
    grbs.close()
    return grb_data, lats, lons, valid_date


def regrid(grb_data, lats, lons):
    """Regrid the data onto an even web mercator grid"""
    logging.info('Regridding data...')
    x, y = webmerc_proj(lats, lons)

    # make new grid
    xn = np.linspace(x.min(), x.max(), grb_data.shape[1])
    yn = np.linspace(y.min(), y.max(), grb_data.shape[0])
    X, Y = np.meshgrid(xn, yn)

    regridded_data = scipy.interpolate.griddata((x.ravel(), y.ravel()),
                                                grb_data.ravel(), (X, Y),
                                                method='linear')
    return regridded_data, X, Y


def save_data(base_dir, valid_date, regridded_data, X, Y):
    """Save the data and grid to a numpy file"""
    logging.info('Saving numpy data to a file...')
    thedir = os.path.join(os.path.expanduser(base_dir),
                          valid_date.strftime('%Y/%m/%d'))
    if not os.path.isdir(thedir):
        os.makedirs(thedir)

    path = os.path.join(thedir, valid_date.strftime('%HZ.npz'))
    np.savez_compressed(path, data=regridded_data, X=X, Y=Y)


def main():
    logging.basicConfig(
        level='WARNING',
        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    argparser = argparse.ArgumentParser(
        description='Retrieve MRMS precipitation data, regrid it, and save',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-v', '--verbose', action='count',
                           help='Increase logging verbosity')
    argparser.add_argument(
        '-d', '--date', default='latest',
        help='Datetime to retrieve data at in ISO format e.g. 2017-07-31T220000Z')  # NOQA
    argparser.add_argument('--save-dir', help='Directory to save data to',
                           default='~/.mrms')
    argparser.add_argument(
        '--bbox', default='31,37,245,257',
        help='The lat/lon bounding box for the data subset like lat0,lat1,lon0,lon1')  # NOQA

    args = argparser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.date != 'latest':
        date = dt.datetime.strptime(args.date, '%Y-%m-%dT%H%M%SZ')
    else:
        date = args.date

    bbox = [float(b) for b in args.bbox.split(',')]

    tmpfile = tempfile.NamedTemporaryFile()
    download_data(date, tmpfile)
    grb_data, lats, lons, valid_date = read_subset(tmpfile.name, bbox)
    regridded_data, X, Y = regrid(grb_data, lats, lons)
    save_data(args.save_dir, valid_date, regridded_data, X, Y)
    tmpfile.close()


if __name__ == '__main__':
    main()
