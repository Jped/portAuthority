from pymongo import MongoClient
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statistics
import pandas as pd
from operator import itemgetter
from itertools import permutations
import matplotlib.dates as mdates
from scipy.misc import factorial
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from lmfit import Model
import calendar
db = MongoClient().PADATA
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

"""
    This function takes in a filename, changes the file from | to , delinated and then runs an os command to import to mongo
"""
def dump_data(filename, gate):
    new_filename=   "{}.json".format(filename)
    data        =   pd.read_csv(filename, sep="|", skiprows=4)
    data["GATE"]=   str(gate)
    data["TIME"]=   data["TIME"].apply(lambda x:{"$date":x})
    with open(new_filename,"w") as f:
        f.write(data.to_json(orient='records',lines=True))
    #new file is the file that is comma delinated instead of pipe. This is done so it can be added to the mongo db.
    os.system("mongo| mongoimport --db PADATA --collection PortAuthority --file {} --type json --headerline &".format(new_filename))
# dump_data("./233_data/1.txt","233")
# dump_data("./233_data/2.txt","233")
# dump_data("./233_data/3.txt","233")
# dump_data("./233_data/4.txt","233")
# dump_data("./233_data/5.txt","233")

"""
    This function takes in a gate  and a time interval and then displays a graph
"""
def graph_frequency(gate, time_interval, days_of_week=[]):
    #This pipe command is the query of mongo, it is a pretty hefty command, it uses a trick is saw online where it:
    # 1. subtracts the time to a common time (here it is start of epoch)
    # 2. subtracts the time from a common epoch, takes is modulo with the interval you want
    # 3. subtracts 2 from 1, based on this number values that are in the same time interval group will have the same response and be grouped together
    # this also orders the results and returns the first time in the interval...

    comparison = datetime.datetime(1970,01,01)
    if not days_of_week:
        pipe    = [{"$match":{"GATE":str(gate)}},{"$group": {"_id":{"$subtract":[{"$subtract":["$TIME",comparison]}, {"$mod":[{"$subtract":["$TIME",comparison]}, 60000 *time_interval]}]},"addresses": { "$push": "$ADDRESS" }, "start":{"$min":"$TIME"}}}, {"$sort":{"_id":1}}]
    else:
        pipe    = [{"$match":{"GATE":str(gate)}},{"$project":{"dayOfWeek":{"$dayOfWeek":"$TIME"}, "TIME":"$TIME","ADDRESS":"$ADDRESS"}}, {"$match":{"dayOfWeek":{"$in":days_of_week}}}, {"$group": {"_id":{"$subtract":[{"$subtract":["$TIME",comparison]}, {"$mod":[{"$subtract":["$TIME",comparison]}, 60000 *time_interval]}]},"addresses": { "$push": "$ADDRESS" }, "start":{"$min":"$TIME"}}}, {"$sort":{"_id":1}}]
    results = db["PortAuthority"].aggregate(pipe)
    print pipe
    frequency_array = []
    start_time      = []
    #here we are going through the results and addding them to arrays to be graphed
    for r in results:
        #IMPORTANT, WE ARE ONLY COUNTING AN ADDRESS ONCE FOR THE TIME INTERVAL, IE IF SHE SHOWS UP FIVE TIMES IN TIMEINTERVAL THAT IS COUNTED ONCE, HOWEVER, IF SHE SHOWS UP LATER THAT IS
        #COUNTED AGAIN...
        count   =   len(set(r["addresses"]))
        frequency_array.append(count)
        start_time.append(r["start"]+ datetime.timedelta(minutes=(time_interval/2)))
    #create a subplot in order to use the xaxis feature...
    print frequency_array
    print start_time
    ax = plt.subplot(111)
    ax.bar(start_time, frequency_array, width=0.01)
    ax.xaxis_date()
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.xticks(rotation="vertical")
    plt.title("Frequency Based on {} Minute Intervals ".format(time_interval))
    plt.savefig("{}.png".format(gate))
    plt.show()

def graph_month(gate,time_interval, start_date, week_num):
    #time interval is the resolution
    plt.figure(figsize=(12,10), dpi=600)
    week2       = start_date + datetime.timedelta(weeks=1)
    comparison  = datetime.datetime(1970,01,01)
    pipe        = [{"$match":{"$and":[{"GATE":str(gate)},{'TIME':{'$gte':start_date, '$lt': week2}}]}},{"$group":{"_id":"$ADDRESS", "TIME":{"$min":"$TIME"}}}, {'$group': {'count': {'$sum': 1}, '_id': {'$subtract': [{'$subtract': ['$TIME', comparison]}, {'$mod': [{'$subtract': ['$TIME', comparison]},  60000 *time_interval]}]}, 'time': {'$min': '$TIME'}}}]
    results     = db["PortAuthority"].aggregate(pipe)
    week1_count = []
    start_time  = []
    for r in results:
        week1_count.append(r["count"])
        start_time.append(r["time"])
    sorted_times= sorted(zip(start_time,week1_count))
    ax1         = plt.subplot(211)
    week1_count =[]
    start_time  =[]
    for x,y in sorted_times:
        week1_count.append(y)
        start_time.append(x)
    ax1.set_xlim([start_date,week2])
    ax1.set_ylim([0,1400])
    ax1.plot_date(start_time, week1_count, "b-")
    ax1.xaxis_date()
    week3       = week2 + datetime.timedelta(weeks=1)
    pipe        = [{"$match":{"$and":[{"GATE":str(gate)},{'TIME':{'$gte':week2, '$lt': week3}}]}},{"$group":{"_id":"$ADDRESS", "TIME":{"$min":"$TIME"}}}, {'$group': {'count': {'$sum': 1}, '_id': {'$subtract': [{'$subtract': ['$TIME', comparison]}, {'$mod': [{'$subtract': ['$TIME', comparison]},  60000 *time_interval]}]}, 'time': {'$min': '$TIME'}}}]
    results     = db["PortAuthority"].aggregate(pipe)
    week2_count = []
    start_time  = []
    for r in results:
        week2_count.append(r["count"])
        start_time.append(r["time"])
    sorted_times= sorted(zip(start_time,week2_count))
    week2_count =[]
    start_time  =[]
    for x,y in sorted_times:
        week2_count.append(y)
        start_time.append(x)
    ax2         =  plt.subplot(212)
    ax2.set_xlim([week2,week3])
    ax2.set_ylim([0,1400])
    ax2.plot(start_time, week2_count,"b-")
    ax2.xaxis_date()
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a:%D"))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a:%D"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_minor_locator(mdates.HourLocator())
    ax1.set_ylabel("Frequency")
    ax2.set_ylabel("Frequency")
    plt.suptitle("Arrivals Rate Gate:{} with a {} minute resolution. Weeks {} and {}".format(gate,time_interval, week_num, week_num+1))
    plt.savefig("rate-{}Month-{}-Gate-{}.png".format(time_interval,start_date.date(),gate))
    plt.show()

def graph_whole_thing(gate):
    #gets the earliest time and the latest time and calls graph_month until the whole thing is done...
    end_time          =     db["PortAuthority"].find({"GATE":str(gate)}).sort([("TIME",-1)]).limit(1)[0]["TIME"]
    start_time        =     db["PortAuthority"].find({"GATE":str(gate)}).sort([("TIME",1)]).limit(1)[0]["TIME"]
    days_of_week      =     start_time.weekday()
    print start_time
    print end_time
    mon_start         =     start_time - datetime.timedelta(days=days_of_week)
    mon_start         =     mon_start.replace(hour=0,minute=0,second=0,microsecond=0)
    week_num          =     1
    while True:
        if  mon_start>end_time:
            break
        graph_month(gate,15, mon_start,week_num)
        mon_start     =    mon_start + datetime.timedelta(weeks=2)
        week_num+=2

#graph_whole_thing("204")
#graph_whole_thing("233")

def get_average_wait(gate):
    pipe        = [{"$match":{"GATE":str(gate)}},{"$group":{"_id":"$ADDRESS", "startTime":{"$min":"$TIME"}, "endTime":{"$max":"$TIME"}}}]
    results     = db["PortAuthority"].aggregate(pipe, allowDiskUse=True)
    wait_times  = np.array([((r["endTime"]-r["startTime"]).total_seconds()/60) for r in results])
    print len(wait_times)
    wait_times  = np.ma.masked_equal(wait_times,0)
    print len(wait_times.compressed())
    plt.hist(wait_times.compressed())
    plt.savefig("waittimes.png", dpi=600)
    # mean_wait   = statistics.mean(wait_times)
    # mode_wait   = statistics.mode(wait_times)
    # median_wait = statistics.median(wait_times)
    # stdev_wait  = statistics.stdev(wait_times)
    # print "Mean wait time (Minutes):{}".format(str(mean_wait))
    # print "Mode wait time (Minutes):{}".format(str(mode_wait))
    # print "Median wait time (Minutes):{}".format(str(median_wait))
    # print "Standard Deviation of wait time (Minutes):{}".format(str(stdev_wait))

def origin_destination():
     o_d        = permutations(db["PortAuthority"].distinct("GATE"),2)
     o_d_dict   = {(str(i[0])+ "->" + str(i[1])):0 for i in o_d}
     pipe       = [{"$group":{"_id":"$ADDRESS", "gate_time":{"$push":{"gate":"$GATE", "time":"$TIME"}}}}]
     results    = db["PortAuthority"].aggregate(pipe, allowDiskUse=True)
     for r in results:
     #looping over the people
         time_path = sorted(r["gate_time"], key=itemgetter('time'))
         if len(time_path)>1:
             origin_gate    = time_path[0]["gate"]
             last_time      = time_path[0]["time"]
             len_gates      =  len(time_path)
             for x in range(1,len_gates):
                 #looping over the person's times
                 p          = time_path[x]
                 nex_gate   = p["gate"]
                 time_delta = p["time"] - last_time
                 last_gate  =  time_path[x-1]["gate"]
                 if last_gate != origin_gate and time_delta>=datetime.timedelta(hours=2):
                     #do magic
                     select          = str(origin_gate)+ "->"+ str(last_gate)
                     o_d_dict[select]= o_d_dict[select] + 1
                     origin_gate    =  nex_gate
                 elif x ==(len_gates-1) and nex_gate != origin_gate:
                     #this is a to catch the last location
                     select     = str(origin_gate)+ "->"+ str(nex_gate)
                     o_d_dict[select]= o_d_dict[select] + 1
                 last_time   =   p["time"]
     print o_d_dict

#origin_destination()


def arrival_rate(gate, time_interval, days_of_week=[]):
    #days of week should be an array of numbers desired 1- sunday 7-saturday
    if not days_of_week:
        count               = len(db["PortAuthority"].find({"GATE":str(gate)}).distinct("ADDRESS"))
        start_time          = db["PortAuthority"].find().sort([("TIME",-1)]).limit(1)[0]["TIME"]
        end_time            = db["PortAuthority"].find().sort([("TIME",1)]).limit(1)[0]["TIME"]
        total_time_minutes  = (start_time - end_time).total_seconds()/60
        arrival_rate        = count/total_time_minutes
        print "Arrival Rate: {} unique people per a minute".format(str(arrival_rate))
    else:
        pipe                =   [{"$match":{"GATE":str(gate)}},{"$group":{"_id":"$ADDRESS", "TIME":{"$min":"$TIME"}}},{"$project":{"dayOfWeek":{"$dayOfWeek":"$TIME"}}}, {"$match":{"dayOfWeek":{"$in":days_of_week}}}]
        results             =   db["PortAuthority"].aggregate(pipe)
        count               =   len(list(results)) #there got to be a better way to do this
        start_time          =   db["PortAuthority"].find().sort([("TIME",-1)]).limit(1)[0]["TIME"]
        end_time            =   db["PortAuthority"].find().sort([("TIME",1)]).limit(1)[0]["TIME"]
        total_weeks         =   int((start_time-end_time).days/7)
        total_time_minutes  =   total_weeks*len(days_of_week)*1440.0
        arrival_rate        =   count/total_time_minutes
        print "Arrival Rate: {} unique people per a minute".format(str(arrival_rate))


def arrival_rate_byDate(gate,time_interval, date):
    #run a query that gets unique addresses on a certain date based by the hour....
    comparison = datetime.datetime(1970,01,01)
    pipe = [{"$match":{"$and":[{"GATE":str(gate)},{'TIME':{'$gte':date, '$lt': date +datetime.timedelta(days=1)}}]}},{"$group":{"_id":"$ADDRESS", "TIME":{"$min":"$TIME"}}}, {'$group': {'count': {'$sum': 1}, '_id': {'$subtract': [{'$subtract': ['$TIME', comparison]}, {'$mod': [{'$subtract': ['$TIME', comparison]},  60000 *time_interval]}]}, 'time': {'$min': '$TIME'}}}]
    results = db["PortAuthority"].aggregate(pipe)
    rate_array = []
    start_time      = []
    #here we are going through the results and addding them to arrays to be graphed
    for r in results:
        count   =   r["count"]/time_interval
        rate_array.append(count)
        start_time.append(r["time"])
    #create a subplot in order to use the xaxis feature...

    ax = plt.subplot(111)
    ax.bar(start_time, rate_array, width=0.007)
    ax.xaxis_date()
    ystart, yend = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(ystart,yend, 5))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(60), interval=time_interval))
    plt.xlabel("Hour of day")
    plt.ylabel("Arrival Rate (people per minute)")
    plt.xticks(rotation="vertical")
    plt.title("Arrival Rate on {} based on {} Minute Intervals ".format(date.date(),time_interval))
    plt.savefig("{},{},{}.png".format(gate, date.date(),time_interval), dpi = 600)
    plt.show()

def arrival_byDate(gate,time_interval, date):
        #run a query that gets unique addresses on a certain date based by the hour....
        comparison = datetime.datetime(1970,01,01)
        #pipe = [{"$match":{"$and":[{"GATE":str(gate)},{'TIME':{'$gte':date, '$lt': date +datetime.timedelta(days=1)}}]}},{"$group":{"_id":"$ADDRESS", "TIME":{"$min":"$TIME"}}}, {'$group': {'count': {'$sum': 1}, '_id': {'$subtract': [{'$subtract': ['$TIME', comparison]}, {'$mod': [{'$subtract': ['$TIME', comparison]},  60000 *time_interval]}]}, 'time': {'$min': '$TIME'}}}]
        pipe    =   [{"$match":{"$and":[{"GATE":str(gate)},{'TIME':{'$gte':date, '$lt': date +datetime.timedelta(days=1)}}]}},{"$group": {"_id":"$ADDRESS", "times":{"$push":"$TIME"}}}]
        results = db["PortAuthority"].aggregate(pipe)
        arrival_count   =   {}
        for r in results:
            t       =   r["times"]
            if len(t)>1:
                times   =   sorted(t)
                #got to count the first and last ones somewhere here...
                float_subtract_arrival  = (times[0]-comparison).total_seconds()
                resolution_value_arrival= float_subtract_arrival - (float_subtract_arrival % float(60*time_interval))
                if resolution_value_arrival in arrival_count:
                    count                           = arrival_count[resolution_value_arrival] + 1
                    arrival_count[resolution_value_arrival]    = count
                else:
                    arrival_count[resolution_value_arrival]=1
                if len(times)>2:
                    for x in range(len(times)-1):
                        diff    =   times[x+1]- times[x]
                        if diff>datetime.timedelta(minutes=120):
                            #count an arrival @[x+1] and an exit @[x]
                            #becareful not to recount endpoints
                            if (x+1)!=(len(times)-1):
                                float_subtract_arrival  = (times[x+1]-comparison).total_seconds()
                                resolution_value_arrival= float_subtract_arrival - (float_subtract_arrival % float(60*time_interval))
                                if resolution_value_arrival in arrival_count:
                                    count                           = arrival_count[resolution_value_arrival] + 1
                                    arrival_count[resolution_value_arrival]    = count
                                else:
                                    arrival_count[resolution_value_arrival]=1
        start_time    =  np.array(arrival_count.keys()) + calendar.timegm(comparison.utctimetuple())
        start_time    = [datetime.datetime.fromtimestamp(x) for x in start_time]
        rate_array    = arrival_count.values()
        ax = plt.subplot(111)
        ax.bar(start_time, rate_array, width=0.007)
        ax.xaxis_date()
        # ystart, yend = ax.get_ylim()
        # ax.yaxis.set_ticks(np.arange(ystart,yend, 5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())
        plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(60), interval=time_interval))
        plt.xlabel("Hour of day")
        plt.ylabel("Arrival (people count per minute)")
        plt.xticks(rotation="vertical")
        plt.title("Arrival Count on {} based on {} Minute Intervals ".format(date.date(),time_interval))
        plt.savefig("Frequency-{},{},{}-rev2.png".format(gate, date.date(),time_interval), dpi = 600)
        plt.show()

#arrival_byDate("south",15, datetime.datetime(2016,06,23))
#arrival_byDate(223,15, datetime.datetime(2016,06,22,0,0,0))

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def poisson_fitting(bins,gate,resolution,values,ty):
    scaled_count                =  np.array(values)
    total_observations          =  len(scaled_count)
    print scaled_count
    print "___________SCALED COUNT _____________"
    #basically what happens here is that the count array is scapled to 0 - bins
    high_count                  = float(max(scaled_count))
    print high_count
    print "____________high count __________"
    scaled_count                = scaled_count/high_count
    scaled_count                = scaled_count*bins
    entries, bin_edges, patches = plt.hist(scaled_count, bins=bins, range=[0, bins], normed=True)
    bin_middles                 = 0.5*(bin_edges[1:] + bin_edges[:-1])
    print entries
    parameters, cov_matrix      = curve_fit(poisson, bin_middles, entries)
    x_plot                      = np.linspace(0, bins, 1000)
    lambda_final                = (parameters/float(bins))*high_count
    p_distb                     = poisson(x_plot, *parameters)
    p_distb_seg                 = int(1000/bins)
    p_distb_seg_mid             = int(p_distb_seg/2)
    expected_p_distb            = []
    len_p_distb                 = len(p_distb)
    while p_distb_seg_mid<len_p_distb:
        expected_p_distb.append(p_distb[p_distb_seg_mid])
        p_distb_seg_mid+=p_distb_seg
    expected_p_distb_back_scaled=   map(int, np.array(expected_p_distb)*total_observations)
    entries_back_scaled         =   map(int, np.array(entries)*total_observations)
    chisq                       =  0
    for x in range(len(entries_back_scaled)):
        expected = (expected_p_distb_back_scaled[x])
        if expected !=0:
            chisq+= ((expected-entries_back_scaled[x])**2)/expected
    print "chisq"
    print chisq
    print "labmba"
    print lambda_final
    plt.plot(x_plot, p_distb, 'r-', lw=2)
    plt.title("{} rate at gate:{} with {} bins and a {} minute resolution".format(ty,gate,bins,resolution))
    plt.figtext(0.99,0.01,"chi squared of {} Labmba of {}".format(chisq, lambda_final), horizontalalignment='right')
    plt.savefig("{}-Poisson-{}-{}-{}-rev2.png".format(ty,gate,bins,resolution), dpi = 600)
    plt.close()


def exit_rate(gate,exclude_days=[],exclude_dates=[], exclude_hours=[]):
        time_interval   = 360
        resolution      = 15
        comparison      = datetime.datetime(1970,01,01)
        start_time      = db["PortAuthority"].find({"GATE":str(gate)}).sort([("TIME",1)]).limit(1)[0]["TIME"]
        end_time        = db["PortAuthority"].find({"GATE":str(gate)}).sort([("TIME",-1)]).limit(1)[0]["TIME"]
        query_interval  = end_time-start_time
        num_intervals   = int(((query_interval).total_seconds()*60)/resolution)
        print "initial intervals {}".format(num_intervals)
        match = [{"GATE":str(gate)}]
        if exclude_dates:
            for date in exclude_dates:
                match.append({"TIME":{"$not":{"$gte":date[0],"$lte":date[1]}}})
                seconds_between_dates   = date[1]
                num_intervals-=(((date[1]-date[0]).total_seconds()*60)/resolution)
                print "after exclude dates {}".format(num_intervals)
        if exclude_hours or exclude_days:
            if exclude_hours:
                match.append({"hour":{"$not":{"$in":exclude_hours}}})
                num_intervals-=len(exclude_hours)*(60/resolution)*(query_interval.days-(len(exclude_days)*(query_interval.days/7)))
                print "after exclude hours {}".format(num_intervals)
            if exclude_days:
                match.append({"days":{"$not":{"$in":exclude_days}}})
                num_intervals-=len(exclude_days)*(1440/resolution)
                print "after exclude_days {}".format(num_intervals)
            pipe        = [{"$project":{"hour":{"$hour":"$TIME"},"days":{"$dayOfWeek":"$TIME"},"ADDRESS":"$ADDRESS","TIME":"$TIME","GATE":"$GATE"}},{"$match":{"$and":match}},{"$group": {"_id":{"$subtract":[{"$subtract":["$TIME",comparison]}, {"$mod":[{"$subtract":["$TIME",comparison]}, 60000 *time_interval]}]},"addresses": { "$push": "$ADDRESS" }, "times":{"$push":"$TIME"}}}, {"$sort":{"_id":1}}]
        else:
            pipe            = [{"$match":{"$and":match}},{"$group": {"_id":{"$subtract":[{"$subtract":["$TIME",comparison]}, {"$mod":[{"$subtract":["$TIME",comparison]}, 60000 *time_interval]}]},"addresses": { "$push": "$ADDRESS" }, "times":{"$push":"$TIME"}}}, {"$sort":{"_id":1}}]
        results         = db["PortAuthority"].aggregate(pipe, allowDiskUse=True)

        #results           = [{"addresses":[0,1,0,2,3,4,5,6,0],"times":[datetime.datetime(2017,07,02,hour=9,minute=15),datetime.datetime(2017,07,02,hour=9,minute=21),datetime.datetime(2017,07,02,hour=9,minute=20),datetime.datetime(2017,07,02,hour=9,minute=30),datetime.datetime(2017,07,02,hour=10,minute=45),datetime.datetime(2017,07,02,hour=11,minute=1),datetime.datetime(2017,07,02,hour=1,minute=15),datetime.datetime(2017,07,02,hour=2,minute=30),datetime.datetime(2017,07,02,hour=5,minute=45)]}]
        time_count      = {}
        c = 0
        for r in results:
            addresses       =   r["addresses"]
            times           =   r["times"]
            address_time    =   {}
            for x in range(len(addresses)):
                address     =   addresses[x]
                time        =   times[x]
                if address in address_time:
                    time_array              =   address_time[address]
                    time_array.append(time)
                    address_time[address]   =   time_array
                else:
                    address_time[address]=[time]
            for address in address_time:
                times       =   address_time[address]
                if len(times)>0:
                    c +=1
                    max_time        = max(times)
                    float_subtract  = (max_time-comparison).total_seconds()
                    resolution_value= float_subtract - (float_subtract % float(60*resolution))
                    if resolution_value in time_count:
                        count                           =   time_count[resolution_value] + 1
                        time_count[resolution_value]    = count
                    else:
                        time_count[resolution_value]=1
        values          =   np.array(time_count.values())
        print c
        print "____________________"
        print num_intervals
        num_zeros_vals  =   int(num_intervals-len(values))
        print num_zeros_vals
        values          =   np.append(values, np.zeros(num_zeros_vals))



#exit_rate(202,exclude_days=[1,7],exclude_dates=[(datetime.datetime(2016,05,02),datetime.datetime(2016,06,20)),((datetime.datetime(2016,07,4)),(datetime.datetime(2016,07,11)))],exclude_hours=[0,1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23])

#get day of week dictionary, used to count what days happened
def get_dow_dict(start_date,end_date):
    dow_dict    =   {}
    for x in range(start_date.toordinal(), end_date.toordinal()):
        weekday =   datetime.datetime.fromordinal(x).weekday() + 1
        if weekday in dow_dict:
            dow_dict[weekday]+=1
        else:
            dow_dict[weekday]=1
    return dow_dict

def arrival_exit_rates(exclude_days=[],exclude_dates=[], exclude_hours=[]):
    resolution      = 15
    reset_interval  = 120
    comparison      = datetime.datetime(1970,01,01)
    match           =  []
    if exclude_dates:
        for x in range(len(exclude_dates)):
            date    =   exclude_dates[x]
            match.append({"TIME":{"$not":{"$gte":date[0],"$lte":date[1]}}})
    if exclude_hours or exclude_days:
        if exclude_hours:
            match.append({"hour":{"$not":{"$in":exclude_hours}}})
        if exclude_days:
            match.append({"days":{"$not":{"$in":exclude_days}}})
        pipe        = [{"$project":{"hour":{"$hour":"$TIME"},"days":{"$dayOfWeek":"$TIME"},"ADDRESS":"$ADDRESS","TIME":"$TIME","GATE":"$GATE"}},{"$match":{"$and":match}},{"$group": {"_id":"$ADDRESS","gate":{"$push":"$GATE"}, "times":{"$push":"$TIME"}}}]
    else:
        pipe            = [{"$match":{"$and":match}},{"$group": {"_id":"$ADDRESS", "gate":{"$push":"$GATE"},"times":{"$push":"$TIME"}}}]
    print pipe
    results         = db["PortAuthority"].aggregate(pipe, allowDiskUse=True)
    arrival_count   =   {"204":{}, "202":{}, "south":{},"north":{},"223":{},"233":{}}
    exit_count      =   {"204":{}, "202":{}, "south":{},"north":{},"223":{},"233":{}}
    # c               =   0
    for r in results:
        # c+=1
        t       =   r["times"]
        gates   =   r["gate"]
        if len(t)>1:
            hold    =   sorted(zip(t,gates))
            times   =   [ x for (x,y) in hold]
            g       =   [ y for (x,y) in hold]
            #got to count the first and last ones somewhere here...
            float_subtract_arrival  = (times[0]-comparison).total_seconds()
            resolution_value_arrival= float_subtract_arrival - (float_subtract_arrival % float(60*resolution))
            if resolution_value_arrival in arrival_count[g[0]]:
                count                           = arrival_count[g[0]][resolution_value_arrival] + 1
                arrival_count[g[0]][resolution_value_arrival]    = count
            else:
                arrival_count[g[0]][resolution_value_arrival]=1
            float_subtract_exit  = (times[-1]-comparison).total_seconds()
            resolution_value_exit= float_subtract_exit - (float_subtract_exit % float(60*resolution))
            if resolution_value_exit in exit_count[g[-1]]:
                count                               = exit_count[g[-1]][resolution_value_exit] + 1
                exit_count[g[-1]][resolution_value_exit]   = count
            else:
                exit_count[g[-1]][resolution_value_exit]=1
            #ask professor for the edge case of only two times where the difference is more than two hours between the times...here i am skipping them
            if len(times)>2:
                for x in range(len(times)-1):
                    diff    =   times[x+1]- times[x]
                    if diff>datetime.timedelta(minutes=reset_interval):
                        #count an arrival @[x+1] and an exit @[x]
                        #becareful not to recount endpoints
                        if (x+1)!=(len(times)-1):
                            float_subtract_arrival  = (times[x+1]-comparison).total_seconds()
                            resolution_value_arrival= float_subtract_arrival - (float_subtract_arrival % float(60*resolution))
                            if resolution_value_arrival in arrival_count[g[x+1]]:
                                count                           = arrival_count[g[x+1]][resolution_value_arrival] + 1
                                arrival_count[g[x+1]][resolution_value_arrival]    = count
                            else:
                                arrival_count[g[x+1]][resolution_value_arrival]=1
                        float_subtract_exit  = (times[x]-comparison).total_seconds()
                        resolution_value_exit= float_subtract_exit - (float_subtract_exit % float(60*resolution))
                        if resolution_value_exit in exit_count[g[x]]:
                            count                               = exit_count[g[x]][resolution_value_exit] + 1
                            exit_count[g[x]][resolution_value_exit]   = count
                        else:
                            exit_count[g[x]][resolution_value_exit]=1
    gates                   =   arrival_count.keys()
    #removing mest up dadta
    gates.remove("north")
    gates.remove("233")
    for gate in gates:
        start_time      = db["PortAuthority"].find({"GATE":str(gate)}).sort([("TIME",1)]).limit(1)[0]["TIME"]
        end_time        = db["PortAuthority"].find({"GATE":str(gate)}).sort([("TIME",-1)]).limit(1)[0]["TIME"]
        query_interval  = end_time-start_time
        dow_dict        = {}
        if exclude_dates:
            for x in range(len(exclude_dates)):
                date    =   exclude_dates[x]
                query_interval_temp =   datetime.timedelta(0)
                if x<=(len(exclude_dates)-2):
                    if exclude_days:
                            new_dow_dict=   get_dow_dict(date[1], exclude_dates[x+1][0])
                            dow_dict    =   { k: dow_dict.get(k, 0) + new_dow_dict.get(k, 0) for k in set(dow_dict) | set(new_dow_dict) }
                    query_interval_temp        +=  (exclude_dates[x+1][0]-date[1])
                    print "after exclude dates {}".format(query_interval_temp.days)
                if query_interval_temp>datetime.timedelta(0):
                    query_interval  =   query_interval_temp
        num_intervals   =   (query_interval.total_seconds()/60)/resolution
        if exclude_hours or exclude_days:
            if exclude_hours:
                num_intervals-=len(exclude_hours)*(60/resolution)*(query_interval.days-(len(exclude_days)*(query_interval.days/7)))
                print "after exclude hours {}".format(num_intervals)
            if exclude_days:
                total_days = 0
                for d in exclude_days:
                    if dow_dict:
                        total_days += dow_dict[d]
                print "total days excluded"
                print total_days
                num_intervals-=total_days*(1440/resolution)
                print "after exclude_days {}".format(num_intervals)
        arrival_values          =   np.array(arrival_count[gate].values())
        exit_values             =   np.array(exit_count[gate].values())
        num_zeros_vals          =   int(num_intervals-len(arrival_values))
        num_zeros_vals_exit     =   int(num_intervals-len(exit_values))
        arrival_values          =   np.append(arrival_values, np.zeros(num_zeros_vals))
        exit_values             =   np.append(exit_values, np.zeros(num_zeros_vals_exit))
        poisson_fitting(40,gate,resolution,arrival_values,"arrival")
        poisson_fitting(40,gate,resolution,exit_values, "exit")

#arrival_exit_rates("north",exclude_days=[1,7],exclude_dates=[(datetime.datetime(2015,05,02),datetime.datetime(2016,06,20)),((datetime.datetime(2016,07,04)),(datetime.datetime(2017,07,11)))],exclude_hours=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23])
arrival_exit_rates(exclude_days=[1,2,3,4,5,7],exclude_dates=[(datetime.datetime(2000,05,10),datetime.datetime(2016,06,20)),((datetime.datetime(2016,07,11)),(datetime.datetime(2017,07,11)))],exclude_hours=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

# arrival_rate_byDate(223, 60, datetime.datetime(2016,05,9))
#arrival_rate(223,10)
#arrival_rate_byDate(223, 15, datetime.datetime(2016,05,03))
#graph_month(202,15, datetime.datetime(2016,06,04))
#poisson_fitting(40,223,exclude_days=[1,7],exclude_dates=[(datetime.datetime(2000,05,10),datetime.datetime(2016,06,20)),((datetime.datetime(2016,07,04)),(datetime.datetime(2017,07,11)))],exclude_hours=[0,1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23])
