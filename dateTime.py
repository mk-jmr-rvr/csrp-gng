from datetime import datetime
import pytz
import calendar

#failed attempt, but don't delete bc it might get added in the future

def getTime():
    myTimeZone = "Singapore"  
    now = datetime.now(pytz.timezone(myTimeZone))
    mm = str(now.month)
    dd = str(now.day)
    yyyy = str(now.year)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    if now.minute < 10:
        minute = '0' + str(now.minute)
    if now.hour >= 12:
        ampm = ' PM'
    else:
        ampm = ' AM'
    if now.hour > 12:
        hour = str(now.hour - 12)
    weekday = calendar.day_name[now.weekday()]
    time_str = "The time is now " + hour + ":" + minute + " " + ampm + " " + myTimeZone
    print("Time function output:", time_str)
    return time_str

def getDate():
    myTimeZone = "Singapore"  
    now = datetime.now(pytz.timezone(myTimeZone))
    mm = str(now.month)
    dd = str(now.day)
    yyyy = str(now.year)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    weekday = now.weekday()
    week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekdayName = week[weekday]
    date_str = "Today is " + weekdayName + ", " + mm + "/" + dd + "/" + yyyy
    print("Date function output:", date_str)
    return date_str
