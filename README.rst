
innovationdiffusion
===================

Implementation of the Bass Model for innovation diffusion.
Free software: MIT license


Features
--------

* Derive p, q, m values via Nonlinear Least Squares estimation
* Use p, q, and m values to predict at-time and cumulative sales numbers
* Using real and predicted values, show overlaid plots for comparison
* Model summary 

Example
-------
Let's demonstrate a small example on a sales dataset of gaming consoles for the period 2005 - 2017.
 .. code-block::python
    import innovationdiffusion
    import pandas as pd

    xbox = pd.read_excel("xbox_sales.xlsx")

    # Normalize time period values
    time_axis = xbox.index
    time_axis -= time_axis.min()

    y_axis = xbox["Sales"].values

    # Non-cumulative case - size of market is equal to sum of sales numbers for all time periods
    total_market = xbox["Sales"].values.sum()
    p0 = [0.2, 0.2, total_market]

    # Use NLS to derive p, q, and m values from existing data.
    # This may be later used for a look-alike analysis for another innovation
    popt, pcov = innovationdiffusion.nls(time_axis.values, y_axis, False, p0)
    p, q, m = popt

    predicted = innovationdiffusion.predict_values(time_axis.values, p, q, m, False)
    predicted

    # Plot real and predicted values together
    innovationdiffusion.plot(time_axis, xbox["Sales"].values, predicted, legends=["Predicted sales", "Real sales"])

    # Output model summary
    innovationdiffusion.summary(xbox["Sales"].values, predicted, time_axis.values)


Credits
-------

Developed by Esfira Babajanyan.