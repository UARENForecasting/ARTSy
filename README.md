Interactive map the MRMS QPE 24hr precipitation over Arizona and New Mexico.

## Running

Python 3 is required to run the code along with the modules in the
requirements.txt file (probably helpful to use conda). Use the
``download_and_regrid.py`` script to download the data, project to web
mercator, and regrid. Fetching the latest data every hour (probably
after :40 is best based on when the products are available) using a
cronjob might make sense. Then run the Bokeh app with

```python
bokeh serve app.py
```
