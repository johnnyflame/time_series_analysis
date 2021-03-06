# -*- coding: utf-8 -*-

#!/home/johnny/anaconda2/bin/python


"""
Anomoly detector

TODO:

--Set up Google VM with required dependencies
--Deploy it on the cloud using Cron
--Shall we automate the radio name retrieval process? 

"""

import argparse
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from twilio.rest import Client
from ConfigParser import SafeConfigParser
from sklearn import preprocessing
from influxdb import DataFrameClient


config = SafeConfigParser()
config_path = '/home/johnny/Photonic/Anomoly Detection/anomoly_detector_config.ini'
config.read(config_path)

#port_number = 8086

account_sid = config.get('SMS_gateway','account_sid')
auth_token = config.get('SMS_gateway','auth_token')

# Getting a list of radio units from the database
radio_units = []

# The ignore list, loaded from Config
ignore_list = json.loads(config.get('radio_ignore','units'))

receiver_numbers = json.loads(config.get('SMS_gateway','receiver_numbers'))
sender_numbers = config.get('SMS_gateway','sender_numbers')
message = None
SMS_client = Client(account_sid,auth_token)


#hostname = 'localhost'
photonic_host = config.get('database', 'host_address')
admin_user = config.get('database', 'admin_user')
admin_password = config.get('database', 'admin_password')

user = config.get('database', 'user')
password = config.get('database', 'password')
port = config.get('database', 'port')
dbname = config.get('database', 'dbname')


debug = config.get('detector_settings','debug')







def send_SMS(client,receiver,sender,message):

    client.messages.create(
        to=receiver,
        from_=sender,
        body=message
)


def smooth(x, window_len=11, window='flat'):
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

def get_unit_list(client):
    query_content = "SHOW MEASUREMENTS"
    result_list = []

    raw_result = client.query(query_content)
    raw_result = raw_result._raw.values()[0][0].values()[0]

    for unit in raw_result:
        for radio in unit:
            result_list.append(radio)

    return result_list



def make_query(client,radio_unit, inspected_value_type,backdate, interval):

    query_content = "SELECT distinct(\"" + inspected_value_type + "\") FROM " + "\"" + radio_unit + "\" " + \
                    " WHERE time >= now() - " + backdate + " AND \""+ inspected_value_type + \
                    "\" < 200 GROUP BY time(" + interval + ") fill(null)"

    print("Querying database...")
    print "unit: " + radio_unit + "\t timeframe: " + backdate + "\t value inspected: " \
          + inspected_value_type + ""



    raw_result = client.query(query_content)

    # print raw_result

    if len(raw_result) <= 0:
        print "no data sample with the description in the timeframe provided"
        return [],[]

    result = raw_result.get(raw_result.keys()[0])

    # print result
    date_time_indices = result.index
    data = result._get_numeric_data().get_values()

    return data, date_time_indices

def apply_noise_filter(window_size, data, indices,mode='flat'):
    smoothed_mean = smooth(data, window_size, mode)
    smoothed_mean_max = 0
    highest_index = 0

    for i in range(len(indices)):
        if smoothed_mean[i] > smoothed_mean_max:
            smoothed_mean_max = smoothed_mean[i]
            highest_index = i

    return smoothed_mean, smoothed_mean_max, highest_index


def main():
    """Instantiate the connection to the InfluxDB client."""

    global radio_units
    client = DataFrameClient(photonic_host, port, user, password, database=dbname)
    # client.switch_user('johnny','johnny')

    radio_units =  get_unit_list(client)

    # Remove the ignore_list from the radio list to perform checks on.
    radio_units = list(set(radio_units) & set(set(radio_units) ^ set(ignore_list)))

    sensitivity = float(config.get('detector_settings','sensitivity'))
    days = int(config.get('detector_settings','time_period'))
    minutes = 10

    backdate = str(days) + "d"
    interval = str(minutes) + "m"
    # sensitivity as a percentage
    value_type = config.get('detector_settings','value_type')

    for i in range(len(radio_units)):
        current_radio = radio_units[i]
        # What if we make this a historical thing since the beginning of data being recorded.
        nh3_data,indices = make_query(client,current_radio,value_type,backdate,interval)

        sensitivity = float(sensitivity)
        available_days = len(nh3_data)

        if available_days > days:
            nh3_data = nh3_data.reshape(len(nh3_data))
            window_size = days


            smoothed_data,max_value,max_index = apply_noise_filter(
                window_size,nh3_data, indices, mode='flat')



            threshold = (1 + sensitivity/100) * smoothed_data[0]

            if smoothed_data[-1] > threshold:
                message = "Alert! Rising trend in " + value_type + "\nUnit: " + current_radio + " is  " \
                "above threshold by " + str(sensitivity) + " percent at: " + str(indices[-1])

                for number in receiver_numbers:
                    send_SMS(client=SMS_client,receiver=number ,sender=sender_numbers,message=message)


                if debug:
                    print ("Max value in the smoothed out result is " + str() + " occuring at: " + str(indices[max_index]) + "\n")
                    plt.plot(nh3_data)
                    plt.plot(smoothed_data)
                    plt.title("Radio unit: " + current_radio + ": " + value_type + " over: " +
                              str(days) + " days, displayed every " + interval + "\ntotal entries: " + str(len(nh3_data))   )
                    plt.show()

        else:
            print "insufficient data in the specified period for unit: " + current_radio + "\n"
            # print "specified period: " + str(days) + " days, when there are only " \
            #       + str(len(nh3_data)) + " entries recorded for unit " + current_radio




if __name__ == '__main__':
    main()