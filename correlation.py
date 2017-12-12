# -*- coding: utf-8 -*-
"""Tutorial for using pandas and the InfluxDB client."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing


from influxdb import DataFrameClient



port_number = 8086

# def connect_to_influxDB(host_address,port_number,user,user_password,database,protocol='json'):
#
#     return DataFrameClient(host=host_address,
#                            port=port_number,username=user,password=user_password,database=database)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(' + str(window_len) + ')')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def local_maximum(x):
    index_array = range(len(x))
    f = np.r_[True, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], True]
    return (index_array[f], x[f])

def make_query(client,radio_unit, inspected_value_type,backdate,threshold="200"):

    query_content = "SELECT distinct(\"" + inspected_value_type + "\") FROM " + "\"" + radio_unit + "\" " + \
                    " WHERE time >= now() - " + backdate + " AND \""+ inspected_value_type + \
                    "\" <"+ threshold + " GROUP BY time(15m) fill(null)"

    print("Querying database...")

    raw_result = client.query(query_content)
    # print raw_result

    if len(raw_result) <= 0:
        print "no data sample with the description in the timeframe provided"
        return [], []

    result = raw_result.get(raw_result.keys()[0])
    # print result
    date_time_indices = result.index
    data = result._get_numeric_data().get_values()

    return data, date_time_indices

def apply_noise_filter(window_size, data, indices, threshold,mode='flat'):
    smoothed_mean = smooth(data, window_size, mode)
    smoothed_mean_max = 0

    for i in range(len(indices)):
        if smoothed_mean[i] > smoothed_mean_max:
            smoothed_mean_max = smoothed_mean[i]
        # if smoothed_mean[i] > threshold:
        #     print "Alert! Level above threshold at: " + str(indices[i])

    print ("Max value in the post-filter results: ", smoothed_mean_max)
    return smoothed_mean



def sample_selection(x_index,y_index,x,y,sampling_size):
    """
    Select samples from 2 attributes.
    :param x_index: 
    :param y_index: 
    :param x: 
    :param y: 
    :param sampling_size: 
    :return: 
    """
    min_length = len(x_index) if len(x_index) < len(y_index) else len(y_index)
    accuracy_threshold = 6
    counter = 0
    sample_indices = []
    output_x = []
    output_y = []

    while counter < sampling_size:
        i = np.random.randint(0,min_length)
        if abs(x_index[i].hour - y_index[i].hour) < accuracy_threshold and i not in sample_indices:
            sample_indices.append(i)
            counter+=1

    for i in sample_indices:
        output_x.append(x[i])
        output_y.append(y[i])

    return output_x,output_y





def plot_correlation(client,current_radio,y_value,x_value,days):

    backdate = str(days) + "d"

    window_size = days
    threshold = 45

    sample_size = 200
    y_data,y_indices = make_query(client,current_radio,y_value,backdate)
    x_data,x_indices = make_query(client, current_radio, x_value, backdate,threshold="99999")


    if len(y_data) < days or len(x_data) < days:
        print "insufficient data"
    else:



        # plt.plot(x_data)
        # plt.title(current_radio +"  " + x_value)
        # plt.show()
        #
        #
        # plt.plot(y_data)
        # plt.title(current_radio +"  " + y_value)
        # plt.show()

        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].plot(x_data)
        axarr[0, 0].set_title(x_value)
        axarr[0, 1].plot(y_data)
        axarr[0, 1].set_title(y_value)

        #
        # for ax in axarr.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')
        # # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axarr.flat:
        #     ax.label_outer()








        y_data = y_data.reshape(len(y_data))
        x_data = x_data.reshape(len(x_data))







        # temperature_data = preprocessing.scale(temperature_data)








        smoothed_y = apply_noise_filter(window_size, y_data, y_indices, threshold, mode='flat')
        smoothed_x = apply_noise_filter(window_size, x_data, x_indices, threshold, mode='flat')

        smoothed_y, smoothed_x = sample_selection(y_indices, x_indices,smoothed_y,smoothed_x,sample_size)

        # plt.plot(nh3_sample, temp_sample, 'go')
        smoothed_y = preprocessing.scale(smoothed_y)
        smoothed_x = preprocessing.scale(smoothed_x)


        # plt.plot(smoothed_x,smoothed_y, 'ro')
        #
        # plt.ylabel(y_value)
        # plt.xlabel(x_value)



        axarr[1, 0].plot(smoothed_x,smoothed_y,'ro')
        # axarr[1, 0].set_title(x_value + " against " + y_value)
        axarr[1, 0].set(xlabel=x_value, ylabel=y_value)

        # for ax in axarr.flat:
        #     ax.set(xlabel=x_value, ylabel=y_value)
        # # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axarr.flat:
        #     ax.label_outer()



        # axarr[1, 0].ylabel(y_value)
        # axarr[1, 0].xlabel(x_value)

        # axarr[1, 1].scatter(x, y ** 2)
        # axarr[1, 1].set_title('Axis [1,1]')

        # plt.plot(smoothed_nh3_data)
        plt.suptitle("Unit: " + current_radio + " values: " + x_value + " and " + y_value)


        # client.switch_database('photonic')
        # print client.get_list_users()



def main():
    """Instantiate the connection to the InfluxDB client."""

    hostname = 'localhost'
    photonic_host = '35.189.3.224'
    port = 8086

    admin_user = 'shane@photonicinnovations.com'
    admin_password = 'CA0inh#16'

    user = 'johnny'
    password = 'johnny'
    dbname = 'photonic'


    # Temporarily avoid line protocol time conversion issues #412, #426, #431.
    protocol = 'json'

    client = DataFrameClient(photonic_host, port, user, password, database=dbname)
    # client.switch_user('johnny','johnny')



    """
    
    Units without sufficient data
    
   
    """

    radio_units = [
                #"2C2FF2",
                # "2C3195",
                #"2C31B0",
                # "2C31BE",
                # "2C31D2",
                # "2C3074",
                # "2BEF08",
                # "2BFA35",
                # "2BFD0E",
                # "2C0115",
                # "2C308B",
                "2C31B4"
    ]

    """
    
    lat lng 
    
    """

    values_types =[
        "ctemp", "hum", "res", "rssi", "rxp", "snr", "tmp"
]

    current_radio = radio_units[0]
    y_value = "nh3"
    x_value = "hum"
    days = 90


    for i in range(len(radio_units)):
        for j in range(len(values_types)):
            plot_correlation(client,radio_units[i],y_value,values_types[j],days)

    plt.show()
def parse_args():
    """Parse the args from main."""
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='localhost',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    main()