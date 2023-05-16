import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def bass_model(t, p, q, m):
    """ Calculate non-cumulative sales values based on t (period), p, q and m (market potential) values.
        Parameters:
            t (int): Point in time
            p (float): Coefficient of innovation
            q (float): Coefficient of imitation
            m (int): Market potential at time t
        
        Returns:
            Predicted value of sales at time t.
    """ # noqa
    return m*(((p+q)**2/p)*np.exp(-(p+q)*t))/(1+(q/p)*np.exp(-(p+q)*t))**2


def bass_model_cumulative(t, p, q, m):
    """ Calculate cumulative sales values based on t (period), p, q and m (market potential) values.
        Parameters:
            t (int): Point in time
            p (float): Coefficient of innovation
            q (float): Coefficient of imitation
            m (int): Market potential at time t
        
        Returns:
            Predicted value of cumulative sales at time t.
    """ # noqa
    return m*((1 - np.exp(-(p+q)*t)) / (1 + (q/p)*np.exp(-(p+q)*t)))


def nls(xdata, ydata, cumulative=False, p0=None):
    """ Use Non-Linear Squares to fit optimal p, q, and m values to existing sales data.
        Parameters:
            xdata (array-like): Time values (X axis)
            ydata (array-like): Sales values (Y Axis)
            cumulative (boolean): Denotes whether cumulative or at-time sales values are used. Defaults to False
            p0 (tuple): Tuple of initial p, q, and m values. Default value is None.

        Returns:
            Tuple of optimal values and covariances for p, q, and m.
    """ # noqa
    f = bass_model_cumulative if cumulative else bass_model
    popt, pcov = curve_fit(f, xdata, ydata, p0)
    return popt, pcov


def predict_values(time_series, p, q, m, cumulative=False):
    """ Predict non-cumulative sales values for the entire time axis with given p, q, and m values.
        Parameters: 
            time_series (array-like): Time periods (X axis)
            p (float): Coefficient of innovation
            q (float): Coefficient of imitation
            m (int): Market potential at time t
            cumulative (boolean): Denotes whether cumulative or at-time sales values are used. Defaults to False

        Returns:
            Series of predicted values for all time periods.
    """ # noqa
    input_values = 4
    output_values = 1
    f = bass_model_cumulative if cumulative else bass_model
    bass_model_ufunc = np.frompyfunc(f, input_values, output_values)
    predicted = bass_model_ufunc(time_series, p, q, m)
    return predicted


def plot(time_axis, actual_values, predicted_values, legends=None, xlabel="Time period", ylabel="Sales", bar_color="darkblue", line_color="red", figsize=(10, 8)): # noqa
    """ Plot real and predicted sales data. 
        Parameters:
            time_axis (array-like): Time values (X axis)
            actual_values (array-like): Actual sales values (Y axis, bar chart)
            predicted_values (array-like): Predicted values (Y axis, line chart)
            legends (array of strings): Legend texts. Defaults to None
            xlabel (string): Label of X axis. Defaults to "Time period".
            ylabel (string): Label of Y axis. Defaults to "Sales"
            bar_color (string): Color of bar chart for actual values. Defaults to "darkblue"
            line_color (string): Color of line chart for predicted values. Defaults to "red"
            figsize (tuple of ints): Size of plot canvas. Defaults to (10, 8)
    """ # noqa
    plt.figure(figsize=figsize)
    plt.bar(time_axis, actual_values, color=bar_color)
    plt.plot(time_axis, predicted_values, color=line_color, linewidth=3)
    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def summary(real_values, predicted_values, time_axis):
    """ Return a summary of the model. 
        Parameters:
            real_values (array-like): Real values (Y axis)
            predicted_values (array-like): Predicted values (Y axis)
            time_axis (array-like): Time values (X axis)
        
        Returns:
            Model summary.
    """ # noqa
    residuals = real_values - predicted_values
    model = sm.OLS(residuals.astype("float"), sm.add_constant(time_axis))
    results = model.fit()
    return results.summary()
