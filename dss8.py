
import os

import dataiku
import pandas as pd, numpy as np

from dataiku import pandasutils as pdu
import time
import random

from math import ceil
from datetime import datetime, timedelta
import json
import random
from collections import OrderedDict

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource, Range1d, Div, HoverTool
from bokeh.models.widgets import Slider, TextInput, DataTable, DateFormatter, TableColumn
from bokeh.plotting import figure

# Instructions to perform simulation
instructions_0 = Div(text="""<b>Instructions to Simulate the Supplement Trend</b>""", width=600)
instructions_1 = Div(text="""Enter a valid Flight Key, Ex Format 01NOV22/BHXAYT/LS1239""", width=600)
instructions_2 = Div(text="""Note: The Flight Key can be found in OWS Dashboard <a href="https://insight.jet2.com/t/ProjectInsight/views/OWS_Daily_Dashboard/OWS_Dashboard" target="_blank">here</a> """, width=600)


handle = dataiku.Folder("ows_rule")
with handle.get_download_stream("template.json") as f:
    rules_dict = json.load(f)

reference_data_ows_api = dataiku.Dataset("reference_data_ows_api")
reference_data = reference_data_ows_api.get_dataframe()
reference_data.drop(columns=['BookingDate'], inplace=True)

# OWS Classes
class DefaultOneWaySurchargeCalculator():
    """
    """
    def __init__(
        self,
        rules_dictionary
        ):
        """
        Args:
            rules_dictionary (dict): A nested dictionary object containing rules for OWS project
        """
        self.rules_dictionary = rules_dictionary

    def get_default_surcharge_percentage(
        self,
        days_to_depart,
        ):
        """ Function to get the default surcharge based on booking date and flight departure date
        Args:
            days_to_depart (int): Difference in days between booking date and departure date

        Returns:
            default_surcharge_percentage (float): Value between 0-1
            default_rule (str): Applicable default rule
        """
        default_rule = self.rules_dictionary['default_rules']
        months_key = self._convert_days_to_depart_months_to_depart_range(days_to_depart)

        # Month wise surcharge percentage
        default_surcharge_dict = {
            '0-1':0.0,
            '1-2':default_rule.get('1-2_months'),
            '2-3':default_rule.get('2-3_months'),
            '3-4':default_rule.get('3-4_months'),
            '4-5':default_rule.get('4-5_months'),
            '5-6':default_rule.get('5-6_months'),
            '6-7':default_rule.get('6-7_months'),
            '7-8':default_rule.get('7-8_months'),
            '8-9':default_rule.get('8-9_months'),
            '9-10':default_rule.get('9-10_months'),
            '10-11':default_rule.get('10-11_months'),
            '11-12':default_rule.get('11-12_months'),
            '12-13':default_rule.get('12-13_months'),
            '13-14':default_rule.get('13-14_months'),
            '14-15':default_rule.get('14-15_months'),
            '15-16':default_rule.get('15-16_months'),
            '16-17':default_rule.get('16-17_months'),
            '17-18':default_rule.get('17-18_months'),
            '18-19':default_rule.get('18-19_months'),
            '19-20':default_rule.get('19-20_months'),
            '20-21':default_rule.get('20-21_months'),
            '21-22':default_rule.get('21-22_months'),
            '22-23':default_rule.get('22-23_months'),
            '23-24':default_rule.get('23-24_months')
        }

        default_surcharge_percentage = default_surcharge_dict.get(months_key, 0.0)
        default_rule = str(months_key) + '_months'
        if days_to_depart<31:
            default_surcharge_percentage, default_rule = self._get_default_surcharge_for_one_month_booking(days_to_depart)
        return default_surcharge_percentage, default_rule


    def _get_default_surcharge_for_one_month_booking(
        self,
        days_to_depart
        ) :
        """ An internal function to get the default surcharge percentage
            when bookings are done within 1 month prior of departure date
        Args:
            days_to_depart (int): Difference in days between booking date and departure date

        Returns:
            default_surcharge_percentage (float): Value between 0-1
            default_rule (str): Applicable default rule
        """
        default_rule = self.rules_dictionary['default_rules']

        # Day Wise surcharge percentage
        if days_to_depart<30:
            if days_to_depart>21:
                default_surcharge_percentage = default_rule.get('22-30_days')
                default_rule = '22-30_days'
            elif days_to_depart>14:
                default_surcharge_percentage = default_rule.get('15-21_days')
                default_rule = '15-21_days'
            elif days_to_depart>7:
                default_surcharge_percentage = default_rule.get('8-14_days')
                default_rule = '8-14_days'
            else:
                default_surcharge_percentage = default_rule.get('0-7_days')
                default_rule = '0-7_days'
        else:
            default_surcharge_percentage = None
            default_rule = 'Error - Booking outside of 1 month range'

        return default_surcharge_percentage, default_rule


    def _convert_days_to_depart_months_to_depart_range(self, days_to_depart):
        """Function to converts the day value to month range considering a month consist of 30 days

        Args:
            days_to_depart (int): Difference in days between booking date and departure date

        Returns:
            months_to_depart_range (str): A string representation of day value as months range

        Example Usage within class:
        >>> self._convert_days_to_depart_months_to_depart_range(10)
        '0-1'

        """
        months_to_depart_range = str(int(ceil(days_to_depart/30)))+'-'+str(int(ceil(days_to_depart/30+1)))

        return months_to_depart_range

class OneWaySurchargeCalculator(DefaultOneWaySurchargeCalculator):
    """
    """
    def calculate_surcharge_percentage(
        self,
        row
        ) :
        """ This function calculates the surcharge percentage based on the details available in the
            booking instance and the derived rule
        Args:
            row (pd.Series) : pandas series representing a single flight ticket booking instance

        Returns:
            row (pd.Series): pandas series object representing a single flight ticket booking
                             instance updated with the surcharge percentage and the details of
                             which rules are applied for calculating the surcharge

        Note:
            Surcharge will be calculated only for bookings with BookingType as Third Party or
            Everything Else and Channel as Internet User and Booking Type as One Way
            A surcharge of 0 will be returned of booking criteria is other than mentioned above

        """
        # Booking Type Details
        BookingType = row.get('BookingType')
        Channel = row.get('Channel')
        ReturnSingle = row.get('ReturnSingle')
        empty_leg = row.get('empty_leg', 0)
        semi_empty_leg = row.get('semi_empty_leg', 0)

        month_mapping = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July",
                        8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
        day_mapping = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday",
                      6:"Sunday"}

        booking_type_list = ['Third Party', 'Everything Else']

        # Create rules columns and assign default values
        row = self._create_rule_columns(row)

        # Logic to wave off surcharge due to Covid
        dept_dt = pd.to_datetime(row['DEPARTUREDATE']).date()
        Covid = 0
        covid_expected_end_date = self.rules_dictionary['additional_rules']['relaxation']['covid_expected_end_date']
        if dept_dt <= pd.to_datetime(covid_expected_end_date).date():
            Covid = 1
            row['default_rule'] = 'Covid'
        elif empty_leg==1:
            row['default_rule'] = 'Empty Leg'
        elif semi_empty_leg==1:
            row['default_rule'] = 'Semi Empty Leg'

        # Check if input prepared booking instance has all the required fields
        if ((BookingType in booking_type_list)&(Channel=='Internet User')&(ReturnSingle=='S')
            &(empty_leg==0)&(semi_empty_leg==0)&(Covid==0)):

            # Get additional fields
            days_to_depart = (pd.to_datetime(row['DEPARTUREDATE']).date()
                                  - pd.to_datetime(row['BookingDate']).date()).days

            arrival_airport = row.get("ROUTE")[3:] #row.get("ArrivalAirport")
            route_type = row.get('route_type')
            departure_month = row.get('DEPARTUREMONTH')
            departure_day = row.get('DEPARTUREDOW')
            same_week_flights_count = row.get('same_week_flights_count')
            outbound = 1 if row.get('DIRECTION', 'Outbound')=='Outbound' else 0
            event = row.get('event', 'No Event')

            same_day_flight_count = row.get('same_day_flights_count', 1)
            expected_holiday_mix = row.get('expected_hols_mix', 0)

            try:
                AIRLINE_FARE = row.get('AIRLINE_FARE')
                Surcharge = row.get('Surcharge')
                if AIRLINE_FARE ==None or AIRLINE_FARE==np.nan:
                    AIRLINE_FARE=100
                if Surcharge ==None or Surcharge==np.nan:
                    Surcharge=0

            except:
                AIRLINE_FARE=100
                Surcharge=0

            current_flight_booking_count = row.get('PASSENGER_COUNT', 1)
            inbound_flights_booking_count = row.get('InboundFlightPassengerAvgCount', 1)
            outbound_flights_booking_count = row.get('OutboundFlightPassengerAvgCount', 1)
            flight_capacity = row.get('FLIGHTCAPACITY', 189)

            try:
                inbooking_imbalance = self._calculate_imbalance(current_flight_booking_count,
                                                        inbound_flights_booking_count, flight_capacity)
            except:
                inbooking_imbalance = 0

            if inbooking_imbalance == None:
                inbooking_imbalance = 0

            try:
                outbooking_imbalance = self._calculate_imbalance(current_flight_booking_count,
                                                        outbound_flights_booking_count, flight_capacity)
            except:
                outbooking_imbalance = 0

            if outbooking_imbalance == None:
                outbooking_imbalance = 0

            #Default Rule - Days To Depart
            surcharge_pct, default_rule = self.get_default_surcharge_percentage(days_to_depart)
            row['default_rule'] = default_rule
            row['default_value'] = surcharge_pct

            rules = self.rules_dictionary['additional_rules']
            # Rule 1 - For one way Outbound Flight Bookings
            if outbound==1:
                ob_rules = rules['outbound']
                depart_month_surcharge_pct = ob_rules["departure_month"].get(
                                                                month_mapping[int(departure_month)])
                surcharge_pct +=depart_month_surcharge_pct
                row['departure_month_rule'] = depart_month_surcharge_pct

                departure_day_surcharge_pct = ob_rules["departure_day"].get(
                                                                    day_mapping[int(departure_day)])
                surcharge_pct +=departure_day_surcharge_pct
                row['departure_day_rule'] = departure_day_surcharge_pct

                same_day_surcharge_pct = ob_rules["same_day_flight_count"].get(
                                                                    str(int(same_day_flight_count)))
                surcharge_pct +=same_day_surcharge_pct
                row['same_day_flight_count_rule'] = same_day_surcharge_pct

                same_week_surcharge_pct = ob_rules["same_week_flight_count"].get(
                                                                str(int(same_week_flights_count)),0)
                surcharge_pct +=same_week_surcharge_pct
                row['same_week_flight_count_rule'] = same_week_surcharge_pct

                if inbooking_imbalance > 0.3:
                    n = ob_rules["inbound_imbalance"].get("0.3_1")
                    inbooking_imbalance_pct = round(n*inbooking_imbalance, 2)
                    surcharge_pct += inbooking_imbalance_pct
                    row['booking_imbalance_rule'] = inbooking_imbalance_pct

            # Rule 2 For one way Inbound Flight Bookings
            elif outbound==0:
                ib_rules = rules['inbound']
                depart_month_surcharge_pct = ib_rules["departure_month"].get(
                                                                month_mapping[int(departure_month)])
                surcharge_pct +=depart_month_surcharge_pct
                row['departure_month_rule'] = depart_month_surcharge_pct

                departure_day_surcharge_pct = ib_rules["departure_day"].get(
                                                                    day_mapping[int(departure_day)])
                surcharge_pct +=departure_day_surcharge_pct
                row['departure_day_rule'] = departure_day_surcharge_pct

                same_day_surcharge_pct = ib_rules["same_day_flight_count"].get(
                                                                    str(int(same_day_flight_count)))
                surcharge_pct +=same_day_surcharge_pct
                row['same_day_flight_count_rule'] = same_day_surcharge_pct

                same_week_surcharge_pct = ib_rules["same_week_flight_count"].get(
                                                                str(int(same_week_flights_count)),0)
                surcharge_pct +=same_week_surcharge_pct
                row['same_week_flight_count_rule'] = same_week_surcharge_pct

                if outbooking_imbalance > 0.3:
                    n = ib_rules["outbound_imbalance"].get("0.3_1")
                    outbooking_imbalance_pct = round(n*outbooking_imbalance, 2)
                    surcharge_pct += outbooking_imbalance_pct
                    row['booking_imbalance_rule'] = outbooking_imbalance_pct

            # Rule 3 (Excpetions and Relaxations)
            if expected_holiday_mix >= rules['holiday_mix']['min_threshold']:
                extra_sur_pct = round(rules['holiday_mix'].get("slope", 0)*expected_holiday_mix,2)
                surcharge_pct += extra_sur_pct
                row['holiday_mix_rule'] = extra_sur_pct

            elif expected_holiday_mix <= rules['relaxation']['lower_holiday_mix_threshold']:
                extra_sur_pct = rules['relaxation'].get("lower_holiday_mix_surcharge_pct", 0)
                surcharge_pct += extra_sur_pct
                row['holiday_mix_rule'] = extra_sur_pct

            if arrival_airport in rules['frequently_visited_destinations']['destinations'] and outbound==1:
                extra_sur_pct = rules['frequently_visited_destinations'].get("surcharge", 0)
                surcharge_pct += extra_sur_pct
                row['freq_visited_dest_rule'] = extra_sur_pct

            if event != 'No Event':
                extra_sur_pct = rules['events'].get("all_events")
                surcharge_pct += extra_sur_pct
                row['exception_event_rule'] = extra_sur_pct

            row['recommended_surcharge'] = round(surcharge_pct*row['AIRLINE_FARE'], 1)
            # TBD - Get Realtime Airline Fare
            row['recommended_surcharge_pct'] = round(surcharge_pct, 3)

        else:
            row['recommended_surcharge_pct'] = 0

        return row

    def _create_rule_columns(self, row):
        """Function to create a place holder for surcharge rules and assiggn a default value of 0

        Args:
            row (pd.Series)

        Returns:
            row (pd.Series)
        """
        rule_columns = ['default_value','departure_month_rule', 'departure_day_rule',
                        'same_day_flight_count_rule', 'same_week_flight_count_rule', 'holiday_mix_rule',
                        'booking_imbalance_rule', 'freq_visited_dest_rule', 'exception_event_rule',
                        'recommended_surcharge_pct', 'recommended_surcharge']
        row['default_rule'] = 'default'
        for col in rule_columns:
            row[col] = 0.0
        return row

    def _calculate_imbalance(self, outbound_seats_booked, inbound_seats_booked, flight_capacity):
        """Function to calculate the difference between current outbound flight booked seat count
            and the average of next 15 days inbound flights booked seats count

            The difference is normalized and represented as a value between 1 and -1
            The value of near to 1 indicates that Outbound flight is almost full and
            the next 15 days inbound flights are almost empty - An example to apply higher surcharge
            on one way outbound booking

        Args:
            outbound_seats_booked (int): Current Outbound Flights Booked seats count
            inbound_seats_booked (int): Next 15 days Inbound Flights Booked seats average count
            flight_capacity (int): Current Outbound Flights maximum seat capacity

        Returns:
            imbalance (float): Value between -1 and 1
        """
        outbound_seats_remain = flight_capacity-outbound_seats_booked
        inbound_seats_remain = flight_capacity-inbound_seats_booked

        imbalance = (inbound_seats_remain - outbound_seats_remain)/max(outbound_seats_remain,
                                                                        inbound_seats_remain)

        return imbalance

def prepare_booking_data(booking_data, reference_data):
    """Function to get the additional details for a given booking instance such as holiday flags,
        Inbound Flights avaerage passenger count, days to depart, etc

    Args:
        booking_data (pd.Series):
        reference_data (pd.DataFrame): An aggregated Flights, Past Bookings and Forecast data

    Returns:
        booking_instance_series (pd.Series):

    """
    OutboundKeyKey = booking_data.get('OutboundKeyKey')
    reference_series = reference_data[(reference_data.FLIGHTKEY_REV==OutboundKeyKey)].iloc[0]
    booking_instance_series = booking_data.append(reference_series)

    return booking_instance_series
    
def prepare_booking_test_data(OutboundKeyKey):
    """
    """
    start_date = datetime.now().strftime("%Y-%m-%d")
    departure_date = datetime.strptime(OutboundKeyKey[:7], "%d%b%y").strftime("%Y-%m-%d")
    booking_test_data = {
        "CustomerID": 12345,
        "BookingReference": "abc123",
        "BookingDate": start_date,
        "BookingType": "Everything Else",
        "Currency": "",
        "ReturnSingle": "S",
        "DepartureAirport": "",
        "ArrivalAirport": "",
        "Duration": 99,
        "OutboundFlightNumber": "",
        "OutboundDt": departure_date,
        "OutboundKeyKey": OutboundKeyKey,
        "InboundDt": "",
        "InboundFlightNumber": "",
        "InboundKeyKey": "",
        "AdultCount": 1,
        "ChildCount": 0,
        "InfantCount": 0,
        "Channel": "Internet User",
        "LS1PackageId": "123456",
        "PASSENGER_COUNT": 0,
        "InboundFlightPassengerAvgCount": 0,
        "OutboundFlightPassengerAvgCount": 0
    }
    return booking_test_data

def simulate_booking_data(booking_test_data, load_factor):
    """
    """
    max_seats = 189*load_factor
    PASSENGER_COUNT = 0
    InboundFlightPassengerAvgCount = 0
    current_flight_seats_allowed_per_booking = [0, 1, 2, 3, 4]
    ibob_flight_seats_allowed_per_booking = [0, 1, 2, 3]
    booking_date_range = pd.date_range(start=booking_test_data['BookingDate'], end=booking_test_data['OutboundDt'])
    current_flight_slope = len(current_flight_seats_allowed_per_booking)/len(booking_date_range)
    ibob_flight_slope = len(ibob_flight_seats_allowed_per_booking)/len(booking_date_range)

    current_flight_booking_dict = {}
    ibob_flight_booking_dict = {}
    PASSENGER_COUNT = 0
    while PASSENGER_COUNT <= max_seats:
        booking_index = random.randint(0, len(booking_date_range)-1)
        booking_date = booking_date_range[booking_index]
        seats_booked = int(current_flight_slope*booking_index + random.uniform(0,3))

        current_flight_booking_dict[booking_date] = seats_booked
        ibob_flight_booking_dict[booking_date] = int(ibob_flight_slope*booking_index + random.uniform(0,3))
        PASSENGER_COUNT+=seats_booked

    # Make sure that the dictionary contains the start and end date
    current_flight_booking_dict[booking_date_range[0]] = 0
    current_flight_booking_dict[booking_date_range[-1]] = 0
    ibob_flight_booking_dict[booking_date_range[0]] = 0
    ibob_flight_booking_dict[booking_date_range[-1]] = 0
    
    current_flight_booking_dict = OrderedDict(sorted(current_flight_booking_dict.items()))
    ibob_flight_booking_dict = OrderedDict(sorted(ibob_flight_booking_dict.items()))

    return current_flight_booking_dict, ibob_flight_booking_dict


def calculate_surchrge_for_simulated_bookings(booking_test_data, current_flight_booking_dict, ibob_flight_booking_dict, reference_data):
    """
    """
    surcharge_dict = {}
    current_flight_cum_booking = {}
    ibob_flight_cum_booking = {}
    display_rule_series = ''

    OutboundKeyKey = booking_test_data.get('OutboundKeyKey')
    try:
        reference_series = reference_data[(reference_data.FLIGHTKEY_REV==OutboundKeyKey)].iloc[0]
        error_instruction.text=""
    except IndexError:
        error_instruction.text="Invalid Flight Key"
        error_instruction.style={'color': 'red'}
        return surcharge_dict, current_flight_cum_booking, ibob_flight_cum_booking, display_rule_series
        
    Direction = reference_series.get("DIRECTION")
    PASSENGER_COUNT=reference_series.get("PASSENGER_COUNT")
    InboundFlightPassengerAvgCount = reference_series.get("InboundFlightPassengerAvgCount")
    OutboundFlightPassengerAvgCount = reference_series.get("OutboundFlightPassengerAvgCount")
    
    # Temp Variable i for getting example rules display 
    temp_count_var = 0
    
    for booking_date, seats_booked in current_flight_booking_dict.items():
        booking_test_data["BookingDate"] = booking_date.strftime("%Y-%m-%d")
        PASSENGER_COUNT += seats_booked
        booking_test_data["PASSENGER_COUNT"] = PASSENGER_COUNT

        if Direction=="Outbound":
            InboundFlightPassengerAvgCount += ibob_flight_booking_dict[booking_date]
            booking_test_data["InboundFlightPassengerAvgCount"] = InboundFlightPassengerAvgCount
            ibob_flight_cum_booking[booking_date] = InboundFlightPassengerAvgCount
        elif Direction=="Inbound":
            OutboundFlightPassengerAvgCount += ibob_flight_booking_dict[booking_date]
            booking_test_data["OutboundFlightPassengerAvgCount"] = OutboundFlightPassengerAvgCount
            ibob_flight_cum_booking[booking_date] = OutboundFlightPassengerAvgCount

        booking_test_series = pd.Series(booking_test_data)
        booking_instance = prepare_booking_data(booking_test_series, reference_data)
        try:
            surcharge_pct_calculator = OneWaySurchargeCalculator(rules_dict)
            booking_instance_surcharge = surcharge_pct_calculator.calculate_surcharge_percentage(booking_instance)

            current_flight_cum_booking[booking_date] = PASSENGER_COUNT

            surcharge_dict[booking_date] = booking_instance_surcharge['recommended_surcharge']

            if temp_count_var==0:
                display_rule_series = booking_instance_surcharge
                temp_count_var+=1
        except:
            pass

    return surcharge_dict, current_flight_cum_booking, ibob_flight_cum_booking, display_rule_series


def generate_surcharge_trend(reference_data, OutboundKeyKey, load_factor):
    """
    """
    # Load the Rule Book
    handle = dataiku.Folder("ows_rule")
    with handle.get_download_stream("template.json") as f:
        rules_dict = json.load(f)

    booking_test_data = prepare_booking_test_data(OutboundKeyKey)

    current_flight_booking_dict, ibob_flight_booking_dict = simulate_booking_data(booking_test_data, load_factor)

    surcharge_dict, current_flight_cum_booking, ibob_flight_cum_booking, display_rule_series = calculate_surchrge_for_simulated_bookings(
                        booking_test_data, current_flight_booking_dict, ibob_flight_booking_dict, reference_data)
    
    return surcharge_dict, display_rule_series

# Set up widgets
OutboundKeyKey_w = TextInput(title="Flight Key", value='01NOV22/BHXAYT/LS123')
error_instruction = Div(text=" ")
#load_factor_w = TextInput(title="Expected Load Factor", value="0.0")

OutboundKeyKey = OutboundKeyKey_w.value
#load_factor = float(load_factor_w.value)

# Set the Plot Layout and call the main function to view the trend
x = []
y = []
source = ColumnDataSource(data=dict(x=x, y=y))

def create_rule_table(series):
    """
    """
    DF = pd.DataFrame(series, columns=['values'])

    Columns = [TableColumn(field=Ci, title=Ci) for Ci in DF.columns] # bokeh columns
    plot_height = 40*len(DF)
    plot_height = min(plot_height, 600)
    data_table = DataTable(columns=Columns, source=ColumnDataSource(DF), height=plot_height, width=340)
    
    return data_table

# Set up a Table for Rules
rules_column_names = ['default_rule',
                      'default_value',
                     'departure_month_rule',
                     'departure_day_rule',
                     'same_day_flight_count_rule',
                     'same_week_flight_count_rule',
                     'holiday_mix_rule',
                     'booking_imbalance_rule',
                     'freq_visited_dest_rule',
                     'exception_event_rule',
                     'recommended_surcharge_pct',
                     'recommended_surcharge']
rules_column_values = [0]*len(rules_column_names)
rules_raw_values = [' ']*len(rules_column_names)
ds = dict(rules_names=rules_column_names, rules_raw_values=rules_raw_values, rules_values=rules_column_values)
Columns = [TableColumn(field=Ci, title=Ci) for Ci in ['rules_names', 'rules_raw_values', 'rules_values']] # bokeh columns
plot_height = 40*len(ds['rules_names'])
plot_height = min(plot_height, 350)
data_table_source = ColumnDataSource(data=ds)
data_table_1 = DataTable(columns=Columns, source=data_table_source, height=plot_height, width=600)
table_1_title = Div(text="Rules Table")

# Set up plot
hover = HoverTool(
        tooltips=[
            ("Booking Date : ", "$x{%F}"),
            ("Surcharge : ", "$y")
        ], formatters={'$x': 'datetime'},
    )

plot = figure(plot_height=500, plot_width=500,
              x_axis_type='datetime', title="Supplement Trend",
              tools=["crosshair,pan,reset,save,wheel_zoom",hover])
plot.xaxis.axis_label = 'Booking Date'
plot.yaxis.axis_label = 'Surcharge (GBP)'
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# Set up callbacks
def update_surcharge_trend(attrname, old, new):
    # Get the current Input box values
    OutboundKeyKey = str(OutboundKeyKey_w.value)
    #load_factor = float(load_factor_w.value)
    load_factor = 0.9

    # Generate the new curve
    surcharge_dict, display_rule_series = generate_surcharge_trend(reference_data,
                                                  OutboundKeyKey=OutboundKeyKey,
                                                  load_factor=load_factor)
    surcharge_dict = OrderedDict(sorted(surcharge_dict.items()))

    x = list(surcharge_dict.keys())
    y = list(surcharge_dict.values())

    source.data = dict(x=x, y=y)
    
    display_rule_series.fillna(0, inplace=True)
    rules_column_names = ['default_rule', 'default_value', 'departure_month_rule', 'departure_day_rule',
                          'same_day_flight_count_rule', 'same_week_flight_count_rule', 'holiday_mix_rule',
                          'booking_imbalance_rule', 'freq_visited_dest_rule', 'exception_event_rule',
                          'recommended_surcharge_pct', 'recommended_surcharge']
    
    rules_column_values = [display_rule_series[x] for x in rules_column_names]
    
    raw_values_c_names = ['OutboundDt', 'DIRECTION', 'DEPARTUREMONTH', 'DEPARTUREDOW',
                          'same_day_flights_count', 'same_week_flights_count', 'expected_hols_mix',
                          'booking_imbalance_rule', 'freq_visited_dest_rule', 'exception_event_rule',
                          'recommended_surcharge_pct', 'recommended_surcharge']
    
    rules_raw_values = [display_rule_series[x] for x in raw_values_c_names]
    ds_ = dict(rules_names=rules_column_names,
               rules_raw_values=rules_raw_values,
                rules_values=rules_column_values)
    data_table_source.data = ds_
    plot.title.text = "Supplement Trend for {}".format(OutboundKeyKey)

#for w in [OutboundKeyKey_w, load_factor_w]:
#    w.on_change('value', update_surcharge_trend)
OutboundKeyKey_w.on_change('value', update_surcharge_trend)
# Set up layouts and add to document
#inputs = widgetbox(OutboundKeyKey_w, load_factor_w)
inputs = widgetbox(OutboundKeyKey_w)
# Definition of the rules

def_0 = Div(text="""<b>Definitions of the Rules:</b>""", width=600)
def_1 = Div(text="""1. <b>Default Rule</b> - (Table A) This rule calculates the difference between booking date and
                    departure date, higher the difference higher will be Surcharge""", width=600)
def_2 = Div(text="""2. <b>Departure Month Rule</b> - (Table G) Outbound flights from February to August will have a fixed
                    contribution from this rule towards surcharge (say 2.5% of Airline Fare).
                    Similarly for Inbound Flights from September to January will have fixed contribution.
                    Note: The raw value of this rule starts with 1 meaning flight is in January and 12 meaning flight is in December""", width=600)
def_3 = Div(text="""3. <b>Departure Day Rule</b> - (Table H) Inbound flights on Monday to Wednesday will have a fixed
                    contribution from this rule towards surcharge (say 2.5% of Airline Fare).
                    Similarly Outbound Flights on Thursday to Sunday will have fixed contribution.
                    Note: The raw value of this rule starts with 0 meaning flight on Sunday and 6 meaning flight on Saturday""", width=600)
def_4 = Div(text="""4. <b>Same day flight count Rule</b> - (Table I) For a given route if there is only one flight then this rule will
                    have a positive contribution towards the surcharge.
                    More than 3 flights on the same day for the same route will have a negative contribution""", width=600)
def_5 = Div(text="""5. <b>Same week flight count Rule</b> - (Table J) For a given route if there are less than 5 flights in a given week then this rule will
                    have a positive contribution towards the surcharge.
                    More than 5 flights for the same route during the same week will not have any contribution towards surcharge""", width=600)
def_6 = Div(text="""6. <b>Holiday Mix Rule</b> - (Table D & F) For the current flight, we get the holiday mix ratio for
                    2019 mapped flight and if the holiday mix is higher for the mapped flight then higher will be the surcharge for the current flight.
                    If the holiday mix ratio is less than 0.7 then surcharge contribution from this rule will be 0,
                    else it would be 15% of the holiday mix ratio""", width=600)
def_7 = Div(text="""7. <b>Booking Imbalance Rule</b> - (Table L) For a given outbound flight, we look at the bookings of next
                    few days inbound flights(10 days for Far Sun, 7 days for Sun and Ski, 4 days for City destinations) and 
                    calculate the imbalance in bookings.
                    This imbalance value ranges from -1 to 1. A value closer to 1 means more outbound bookings are done
                    compared to inbound flights and as a result higher would be the contribution towards surcharge.
                    Similarly, for Inbound flights we look at the bookings of previous few days outbound flights.
                    If the value of imbalance ratio is closer to 1 then it means more inbound bookings are done compared to outbound bookings.""", width=600)
def_8 = Div(text="""8. <b>Frequently Visited Destination Rule</b> - (Table C) Certain destinations will always have higher demand.
                    This rule will have a fixed contribution to the surcharge for such destinations""", width=600)
def_9 = Div(text="""9. <b>Exceptional Event Rule</b> - (Table E) During certain events or festivals the passenger demand will be higher.
                    This rule will have a positive contribution towards surcharge for such events.
                    Events like Covid will have negative or zero contribution""", width=600)

curdoc().add_root(column(instructions_0, instructions_1, instructions_2, inputs, 
                         error_instruction, plot, 
                         table_1_title, data_table_1,
                         def_0, def_1, def_2, def_3, def_4, def_5,
                         def_6, def_7, def_8, def_9,
                         width=800))