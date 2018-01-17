# -*- coding: utf-8 -*-
from ....entities import DateTime
from .extractor import Extractor
import re
from datetime import timedelta, date, datetime
from dateutil.relativedelta import relativedelta
import calendar

# Variations of dates that the parser can capture
year_variations = ['year', 'years', 'yrs', 'yr']
day_variations = ['days', 'day']
minute_variations = ['minute', 'minutes', 'mins', 'min']
hour_variations = ['hrs', 'hours', 'hour', 'hr']
week_variations = ['weeks', 'week', 'wks']
month_variations = ['month', 'months']

# Variables used for RegEx Matching
day_names = 'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
day_names += '|mon|tue|wed|thu|fri|sat|sun'
month_names_long = 'january|february|march|april|may|june|july|august|september|october|november|december'
month_names = month_names_long + '|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
day_nearest_names = 'today|yesterday|tomorrow|tonight|tonite'
numbers = "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
                    eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
                    eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
                    ninety|hundred|thousand|first|second|third|forth|fourth|fifth)"
re_dmy = '(' + "|".join(day_variations + minute_variations + year_variations + week_variations + month_variations) + ')'
re_duration = '(before|after|earlier|later|ago|from\snow)'
re_year = "(17|18|19|20)\d{2}|^(17|18|19|20)\d{2}"
re_timeframe = 'current|this|coming|next|following|previous|last|end\sof\sthe|since'
re_ordinal = 'st|nd|rd|th|first|second|third|fourth|fourth|' + re_timeframe
re_number_end = ['st', 'nd', 'rd', 'th']
re_time = '(?P<hour>\d{1,2})(\s?)(\:(?P<minute>\d{1,2})(?P<convention>am|pm))'
re_separator = 'of|at|on'
all_numbers = numbers + '|first|second|third|forth|fourth|fifth|sixth|seventh|eighth|ninth|tenth'

# A list tuple of regular expressions / parser fn to match
# The order of the match in this list matters, So always start with the widest match and narrow it down
regex = [
    # Houndify bug fix
    (re.compile(
        r'''
        (\b)
        (
            ((?P<number_1>\d+|(%s[-\s]?)+))? # Matches any number or string 25 or twenty five
            (\s)?
            (%s\s)?
            (\s)?
            ((%s)\s)?
            (?P<month>%s) # Matches any month name
            (\s)
            ((?P<number_2>\d{1,2}|(%s))\s) # Matches any month name
            (%s\s)?
            ((?P<number_3>\d{1,2}|(%s))) # Matches any month name
            (%s\s)?
        )
        (\b)
        '''%(numbers, re_ordinal, re_separator, month_names, all_numbers, re_ordinal, all_numbers, re_ordinal),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateFromWords(
            base_date,
            m.group('number_1'),
            m.group('month'),
            m.group('number_2'),
            m.group('number_3')
        )
    ),
    (re.compile(
        r'''
        (\b)
        (
            ((?P<dow>%s)[,\s]\s*)? #Matches Monday, 12 Jan 2012, 12 Jan 2012 etc
            (?P<day>\d{1,2}) # Matches a digit
            (%s)?
            [-\s] # One or more space
            ((%s)\s)?
            (?P<month>%s) # Matches any month name
            [-\s] # Space
            (?P<year>%s) # Year
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        (\b)
        '''% (day_names, re_ordinal, re_separator, month_names, re_year, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                hashmonths[m.group('month').strip().lower()],
                int(m.group('day') if m.group('day') else 1),
            ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), getDateUnit(m.group('day'), m.group('month'), m.group('year')))
    ),
    (re.compile(
        r'''
        (
            ((?P<dow>%s)[,\s][-\s]*)? #Matches Monday, Jan 12 2012, Jan 12 2012 etc
            (\b)
            (?P<month>%s) # Matches any month name
            [-\s] # Space
            ((?P<day>\d{1,2})) # Matches a digit
            (\b)
            ([\,])?
            (%s)?
            ([\,])?
            ([-\s](?P<year>%s)) # Year
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        '''% (day_names, month_names, re_ordinal, re_year, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                hashmonths[m.group('month').strip().lower()],
                int(m.group('day') if m.group('day') else 1)
            ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), getDateUnit(m.group('day'), m.group('month'), m.group('year')))
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<month>%s) # Matches any month name
            [-\s] # One or more space
            (?P<day>\d{1,2}) # Matches a digit
            (%s)?
            [-\s]\s*?
            (?P<year>%s) # Year
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        (\b)
        '''% (month_names, re_ordinal, re_year, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                hashmonths[m.group('month').strip().lower()],
                int(m.group('day') if m.group('day') else 1),
            ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), getDateUnit(m.group('day'), m.group('month'), m.group('year')))
    ),
    (re.compile(
        r'''
        (\b)
        (
            ((?P<number>\d+|(%s[-\s]?)+)\s)? # Matches any number or string 25 or twenty five
            (?P<unit>%s)s?\s # Matches days, months, years, weeks, minutes
            (?P<duration>%s) # before, after, earlier, later, ago, from now
            (\s*(?P<base_time>(%s)))?
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        (\b)
        '''% (numbers, re_dmy, re_duration, day_nearest_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (dateFromDuration(
            base_date,
            m.group('number'),
            m.group('unit').lower(),
            m.group('duration').lower(),
            m.group('base_time')
        ) + timedelta(**convertTimetoHourMinute(
            m.group('hour'),
            m.group('minute'),
            m.group('convention')
        )), normalizeUnit(m.group('unit').lower()))
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<duration>%s)\s # before, after, earlier, later, ago, from now
            ((?P<number>\d+|(%s[-\s]?)+)\s)? # Matches any number or string 25 or twenty five
            (?P<unit>%s)s
        )
        (\b)
        '''% (re_duration, numbers, re_dmy),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (dateFromDuration(
            base_date,
            m.group('number'),
            m.group('unit').lower(),
            m.group('duration').lower()
        ), normalizeUnit(m.group('unit').lower()))
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<ordinal>%s) # First quarter of 2014
            \s+
            quarter\sof
            \s+
            (?P<year>%s)
        )
        (\b)
        '''% (re_ordinal, re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateFromQuarter(
            base_date,
            hashordinals[m.group('ordinal').lower()],
            int(m.group('year') if m.group('year') else base.year)
        )
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<ordinal_value>\d+)
            (?P<ordinal>%s) # 1st January 2012
            ((\s|,\s|\s(%s))?\s*)?
            (?P<month>%s)
            ([,\s]\s*(?P<year>%s))?
        )
        (\b)
        '''% (re_ordinal, re_separator, month_names, re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                int(hashmonths[m.group('month').lower()] if m.group('month') else 1),
                int(m.group('ordinal_value') if m.group('ordinal_value') else 1),
            ), 'day')
    ),
    (re.compile(
        r'''
        (\b)
        (
            ((?P<number>\d+|(%s[-\s]?)+)\s) # Matches any number or string 25 or twenty five
            (?P<ordinal>%s)? # 1st January 2012
            (?P<month>%s)
            ([,\s]\s*(?P<year>%s))
        )
        (\b)
        '''% (numbers, re_separator, month_names, re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateFromWordsMonthYear(base_date, m.group('number'), m.group('month'), m.group('year'))
    ),
    (re.compile(
        r'''
        (\b)
        (
            ((?P<number>\d+|(%s[-\s]?)+)\s) # Matches any number or string 25 or twenty five
            (?P<ordinal>%s)? # 27 August 17
            (?P<month>%s)
            ([,\s]\s*(?P<year>\d{2}))
        )
        (\b)
        '''% (numbers, re_separator, month_names),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateFromWordsMonthYear(
            base_date,
            m.group('number'),
            m.group('month'),
            m.group('year')
        )
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<month>%s)
            \s+
            (?P<ordinal_value>\d+)
            (?P<ordinal>%s) # January 1st 2012
            ([,\s]\s*(?P<year>%s))?
        )
        (\b)
        '''% (month_names, re_ordinal, re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                int(hashmonths[m.group('month').lower()] if m.group('month') else 1),
                int(m.group('ordinal_value') if m.group('ordinal_value') else 1),
            )
    ),
    (re.compile(
        r'''
        (\b)
        (?P<time>%s) # this, next, following, previous, last
        \s+
        (?P<dmy>%s) # year, day, week, month, night, minute, min
        ((\s|,\s|\s(%s))?\s*(%s)) # With time
        (\b)
        '''% (re_timeframe, re_dmy, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE),
        ),
        lambda m, base_date: (dateFromRelativeWeekYear(
            base_date,
            m.group('time'),
            m.group('dmy')
        ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), normalizeUnit(m.group('dmy')))
    ),
    (re.compile(
        r'''
        (\b)
        (?P<time>%s) # this, next, following, previous, last => DATE RANGE
        \s+
        ((?P<number>\d+|(%s[-\s]?)+)\s*)?
        (?P<dmy>%s) # year, day, week, month, night, minute, min
        (\b)
        '''% (re_timeframe, numbers, re_dmy),
        (re.VERBOSE | re.IGNORECASE),
        ),
        lambda m, base_date: dateFromRelativeWeekYear(
            base_date,
            m.group('time'),
            m.group('dmy'),
            m.group('number')
        )
    ),
    (re.compile(
        r'''
        (\b)
        (?P<time>%s) # this, next, following, previous, last
        \s+
        (?P<dow>%s) # mon - fri
        ((\s|,\s|\s(%s))?\s*(%s))?
        (\b)
        '''% (re_timeframe, day_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE),
        ),
        lambda m, base_date: (dateFromRelativeDay(
            base_date,
            m.group('time'),
            m.group('dow')
        ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), 'day')
    ),
    (re.compile(
        r'''
        (\b)
        (?P<time>%s) # this, next, following, previous, last (last december)
        \s+
        (?P<month>%s) # Month
        ((\s|,\s|\s(%s))?\s*(%s))?
        (\b)
        '''% (re_timeframe, month_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE),
        ),
        lambda m, base_date: (dateFromRelativeMonth(
            base_date,
            m.group('time'),
            m.group('month')
        ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), 'month')
    ),
    (re.compile(
        r'''
        (\b)
        (
            ((?P<number>\d+|(%s[-\s]?)+)\s) # Matches any number or string 25 or twenty five
            (?P<ordinal>%s)? # 1st January
            (?P<month>%s)
        )
        (\b)
        '''% (numbers, re_separator, month_names),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateFromWordsMonthYear(base_date, m.group('number'), m.group('month'), base_date.year)
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<day>\d{1,2}) # Day, Month
            (%s)?
            [-\s]? # One or more space
            (?P<month>%s)
        )
        (\b)
        '''% (re_ordinal, month_names),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                base_date.year,
                hashmonths[m.group('month').strip().lower()],
                int(m.group('day') if m.group('day') else 1)
            ), 'day')
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<month>%s) # Month, day
            [-\s] # One or more space
            ((?P<day>\d{1,2})\b) # Matches a digit January 12
            (%s)?
        )
        (\b)
        '''% (month_names, re_ordinal),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                base_date.year,
                hashmonths[m.group('month').strip().lower()],
                int(m.group('day') if m.group('day') else 1)
            ), 'day')
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<month>%s) # Month, year
            [-\s] # One or more space
            ((?P<year>\d{1,4})\b) # Matches a digit January 12
        )
        (\b)
        '''% (month_names),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
                int(m.group('year')),
                hashmonths[m.group('month').strip().lower()],
                1
            ), 'month')
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<month>\d{1,2}) # MM/DD or MM/DD/YYYY
            /
            ((?P<day>\d{1,2}))
            (/(?P<year>%s))?
        )
        (\b)
        '''% (re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                int(m.group('month').strip()),
                int(m.group('day'))
            )
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<day>\d{1,2}) # MM/DD or MM/DD/YYYY
            /
            ((?P<month>\d{1,2}))
            (/(?P<year>%s))?
        )
        (\b)
        '''% (re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                int(m.group('month').strip()),
                int(m.group('day'))
            )
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<day>\d{1,2}) # MM-DD-YYYY
            -
            ((?P<month>\d{1,2}))
            -
            ((?P<year>%s))
        )
        (\b)
        '''% (re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: datetime(
                int(m.group('year') if m.group('year') else base_date.year),
                int(m.group('month').strip()),
                int(m.group('day'))
            )
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<year_start>%s)
            (-)
            (?P<year_end>%s)
        )
        (\b)
        '''% (re_year, re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateYearRange(base_date, m.group('year_start'), m.group('year_end'))
    ),
    (re.compile(
        r'''
        (\b)
        (
            (?P<duration>%s) # before, after, earlier, later, ago, from now
            \s
            ((?P<year>%s)|(?P<day>\d{1,2})) # MM/DD or MM/DD/YYYY
        )
        (\b)
        '''% (re_timeframe, re_year),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: dateFromDurationDigit(
                base_date,
                m.group('duration'),
                m.group('year'),
                m.group('day')
            )
    ),
    (re.compile(
        r'''
        (\b)
        (?P<adverb>%s) # today, yesterday, tomorrow, tonight
        ((\s|,\s|\s(%s))?\s*(%s))?
        '''% (day_nearest_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (dateFromAdverb(
            base_date,
            m.group('adverb')
        ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), 'day')
    ),
    (re.compile(
        r'''
        (\b)
        (?P<named_day>%s) # Mon - Sun
        (\b)
        '''% (day_names),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: this_week_day(
            base_date,
            hashweekdays[m.group('named_day').lower()]
        )
    ),
    # Disable year
    # (re.compile(
    #     r'''
    #     (\b)
    #     (?P<year>%s) # Year
    #     (\b)
    #     '''% (re_year),
    #     (re.VERBOSE | re.IGNORECASE)
    #     ),
    #     lambda m, base_date: (datetime(int(m.group('year')), 1, 1), 'year')
    # ),
    (re.compile(
        r'''
        (\b)
        (?P<month>%s) # Month
        (\b)
        '''% (month_names_long),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m, base_date: (datetime(
            base_date.year,
            hashmonths[m.group('month').lower()],
            1
        ), 'month')
    ),
    (re.compile(
        r'''
        (%s) # Matches time 12:00
        '''% (re_time),
        (re.VERBOSE | re.IGNORECASE),
        ),
        lambda m, base_date: (datetime(
            base_date.year,
            base_date.month,
            base_date.day
        ) + timedelta(**convertTimetoHourMinute(
                m.group('hour'),
                m.group('minute'),
                m.group('convention')
            )), 'hour')
    ),
    (re.compile(
        r'''
        (
            (?P<hour>\d+) # Matches 12 hours, 2 hrs
            \s+
            (%s)
        )
        '''% ('|'.join(hour_variations)),
        (re.VERBOSE | re.IGNORECASE),
        ),
        lambda m, base_date: (datetime(
            base_date.year,
            base_date.month,
            base_date.day,
            int(m.group('hour'))
            ), 'hour')
    )
]

# Hash of numbers
# Append more number to modify your match
def hashnum(number):
    if re.match(r'one|^a\b', number, re.IGNORECASE):
        return 1
    if re.match(r'two\b', number, re.IGNORECASE):
        return 2
    if re.match(r'three\b', number, re.IGNORECASE):
        return 3
    if re.match(r'four\b', number, re.IGNORECASE):
        return 4
    if re.match(r'five\b', number, re.IGNORECASE):
        return 5
    if re.match(r'fifth\b', number, re.IGNORECASE):
        return 5
    if re.match(r'six\b', number, re.IGNORECASE):
        return 6
    if re.match(r'seven\b', number, re.IGNORECASE):
        return 7
    if re.match(r'eight\b', number, re.IGNORECASE):
        return 8
    if re.match(r'nine\b', number, re.IGNORECASE):
        return 9
    if re.match(r'ten\b', number, re.IGNORECASE):
        return 10
    if re.match(r'eleven\b', number, re.IGNORECASE):
        return 11
    if re.match(r'twelve\b', number, re.IGNORECASE):
        return 12
    if re.match(r'thirteen\b', number, re.IGNORECASE):
        return 13
    if re.match(r'fourteen\b', number, re.IGNORECASE):
        return 14
    if re.match(r'fifteen\b', number, re.IGNORECASE):
        return 15
    if re.match(r'sixteen\b', number, re.IGNORECASE):
        return 16
    if re.match(r'seventeen\b', number, re.IGNORECASE):
        return 17
    if re.match(r'eighteen\b', number, re.IGNORECASE):
        return 18
    if re.match(r'nineteen\b', number, re.IGNORECASE):
        return 19
    if re.match(r'twenty\b', number, re.IGNORECASE):
        return 20
    if re.match(r'thirty\b', number, re.IGNORECASE):
        return 30
    if re.match(r'forty\b', number, re.IGNORECASE):
        return 40
    if re.match(r'fifty\b', number, re.IGNORECASE):
        return 50
    if re.match(r'sixty\b', number, re.IGNORECASE):
        return 60
    if re.match(r'seventy\b', number, re.IGNORECASE):
        return 70
    if re.match(r'eighty\b', number, re.IGNORECASE):
        return 80
    if re.match(r'ninety\b', number, re.IGNORECASE):
        return 90
    if re.match(r'hundred\b', number, re.IGNORECASE):
        return 100
    if re.match(r'thousand\b', number, re.IGNORECASE):
      return 1000

    return None

def convert_word_to_number (s):
    if hashnum(s):
        return hashnum(s)
    elif s in hashordinals:
        return hashordinals[s]
    return 1

# Convert strings to numbers
def convert_string_to_number(value):
    # print (value)
    # print (re.findall(numbers + '\\b', value, re.IGNORECASE))
    # print (numbers + '\b')
    if value == None:
        return 1
    if isinstance(value, int):
        return value
    if value.isdigit():
        return int(value)
    num_list = map(lambda s:convert_word_to_number(s), re.findall(numbers + '\\b', value, re.IGNORECASE))
    return sum(num_list)

# Convert time to hour, minute
def convertTimetoHourMinute(hour, minute, convention):
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if convention is None:
        convention = 'am'

    hour = int(hour)
    minute = int(minute)

    if convention == 'pm':
        hour+=12

    return { 'hours': hour, 'minutes': minute }

# Date from words
def dateFromWords (base_date, n1, month, n2, n3):
    day = convert_string_to_number(n1)
    month = hashmonths[month.strip().lower()]
    n2 = convert_string_to_number(n2)
    n3 = convert_string_to_number(n3)
    year = int(str(n2) + str(n3)) if n2 and n2 else base_date.year
    return datetime(year, month, day)

def dateFromWords2 (base_date, n,m):
    return datetime(base_date.year, m, convert_string_to_number(n))

def normalize_year (yr, base_date):
    if yr < 100:
        yr = int(str(base_date.year)[0:2] + str(yr))
    return yr

def dateFromWordsMonthYear (base_date, numberAsString, month, year):
    year = normalize_year(int(year if year else base_date.year), base_date)
    month = int(hashmonths[month.lower()] if month else 1)
    num = convert_string_to_number(numberAsString)
    return datetime(year, month, num)

# Quarter of a year
def dateFromQuarter (base_date, ordinal, year):
    interval = 3
    measurement_unit = 'quarter'
    month_start = interval * (ordinal - 1)
    if month_start < 0:
        month_start = 9
    month_end = month_start + interval
    if month_start == 0:
        month_start = 1
    return ([
        datetime(year, month_start, 1),
        datetime(year, month_end, calendar.monthrange(year, month_end)[1])
    ], measurement_unit)

# Converts relative day to time
# this tuesday, last tuesday
def dateFromRelativeDay(base_date, time, dow):
    # Reset date to start of the day
    base_date = datetime(base_date.year, base_date.month, base_date.day)
    # Convert to lower
    if time:
        time = time.lower()
    dow = dow.lower()
    if time == 'this' or time == 'coming' or time == 'current':
        # Else day of week
        num = hashweekdays[dow]
        return this_week_day(base_date, num)
    elif time == 'last' or time == 'previous':
        # Else day of week
        num = hashweekdays[dow]
        return previous_week_day(base_date, num)
    elif time == 'next' or time == 'following':
        # Else day of week
        num = hashweekdays[dow]
        return next_week_day(base_date, num)
  
# Converts relative month to time
# this december, last january
def dateFromRelativeMonth(base_date, time, month):
    # Reset date to start of the day
    base_date = datetime(base_date.year, base_date.month, base_date.day)
    # Convert to lower
    if time:
        time = time.lower()
    month = month.lower()
    if time == 'this' or time == 'coming' or time == 'current':
        # Else day of week
        num = hashmonths[month]
        return datetime(base_date.year, num, 1)
    elif time == 'last' or time == 'previous':
        # Else day of week
        num = hashmonths[month]
        return datetime(base_date.year, num, 1) + relativedelta(years=-1)
    elif time == 'next' or time == 'following':
        # Else day of week
        num = hashmonths[month]
        return datetime(base_date.year, num, 1) + relativedelta(years=1)

# Converts relative day to time
# this tuesday, last tuesday
def dateFromRelativeWeekYear(base_date, time, dow, ordinal = None):
    # If there is an ordinal (next 3 weeks) => return a start and end range
    # Reset date to start of the day
    # ordinal = int(ordinal) if ordinal is not None and ordinal.isdigit() else 1
    # Convert to lower
    measurement_unit = 'day'
    if time:
        time = time.lower()
    if ordinal is not None:
        ordinal = ordinal.strip()
    if ordinal is not None and ordinal.isdigit():
        ordinal = int(ordinal)
    else:
        if ordinal in hashordinals:
            ordinal = hashordinals[ordinal]
        else:
            ordinal = 1
    d = datetime(base_date.year, base_date.month, base_date.day)
    if dow in year_variations:
        measurement_unit = 'year'
        if time == 'this' or time == 'coming' or time == 'current':
            return (datetime(d.year, 1, 1), measurement_unit)
        elif time == 'last' or time == 'previous':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(datetime(d.year, d.month, 1) + relativedelta(years=-ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (datetime(d.year - ordinal, d.month, 1), measurement_unit)
        elif time == 'next' or time == 'following':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(datetime(d.year, d.month, 1) + relativedelta(years=+ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (datetime(d.year + 1, d.month, 1), measurement_unit)
        elif time == 'end of the':
            return (datetime(d.year, 12, 31), measurement_unit)
    elif dow in month_variations:
        measurement_unit = 'month'
        if time == 'this' or time == 'current':
            return (datetime(d.year, d.month, d.day), measurement_unit)
        elif time == 'last' or time == 'previous':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(datetime(d.year, d.month, d.day) + relativedelta(months=-ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (datetime(d.year, d.month, d.day) + relativedelta(months=-1), measurement_unit)
        elif time == 'next' or time == 'following':
            return (datetime(d.year, d.month, d.day) + relativedelta(months=1), measurement_unit)
        elif time == 'end of the':
            return (datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1]), measurement_unit)
    elif dow in week_variations:
        measurement_unit = 'week'
        if time == 'this' or time == 'current':
            return (d - timedelta(days=d.weekday()), measurement_unit)
        elif time == 'last' or time == 'previous':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(d - timedelta(weeks=ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (d - timedelta(weeks=1), measurement_unit)
        elif time == 'next' or time == 'following':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(d + timedelta(weeks=ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (d + timedelta(weeks=1), measurement_unit)
        elif time == 'end of the':
            day_of_week = base_date.weekday()
            return (d + timedelta(days=6 - d.weekday()), measurement_unit)
    elif dow in day_variations:
        if time == 'this' or time == 'current':
            return (d, measurement_unit)
        elif time == 'last' or time == 'previous':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(d - timedelta(days=ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (d - timedelta(days=1), measurement_unit)
        elif time == 'next' or time == 'following':
            if ordinal > 1:
                values = []
                while ordinal > 0:
                    values.append(d + timedelta(days=ordinal))
                    ordinal = ordinal - 1
                return (values, measurement_unit)
            return (d + timedelta(days=1), measurement_unit)
        elif time == 'end of the':
            return (datetime(d.year, d.month, d.day, 23, 59, 59), measurement_unit)

# Convert Day adverbs to dates
# Tomorrow => Date
# Today => Date
def dateFromAdverb(base_date, name):
    measurement_unit = 'day'
    # Reset date to start of the day
    d = datetime(base_date.year, base_date.month, base_date.day)
    if name:
        name = name.lower()
    if name == 'today' or name == 'tonite' or name == 'tonight':
        return d.today()
    elif name == 'yesterday':
        return d - timedelta(days=1)
    elif name == 'tomorrow' or name == 'tom':
        return d + timedelta(days=1)

def dateYearRange(base_date, year_start, year_end):
    values = []
    year_start = int(year_start)
    year_end = int(year_end)
    if year_start > year_end:
        tmp = year_start
        year_start = year_end
        year_end = tmp
    for i in range(year_start, year_end + 1):
        values.append(datetime(i, 1, 1))
    return (values, 'year')

def getDateUnit (day, year, month):
  if day is not None:
    return 'day'
  if month is not None:
    return 'month'
  if year is not None:
    return 'year'

def normalizeUnit (unit):
  if unit in year_variations:
      return 'year'
  if unit in month_variations:
      return 'month'
  if unit in week_variations:
      return 'week'
  if unit in day_variations:
      return 'day'
  if unit in hour_variations:
      return 'hour'
  if unit in minute_variations:
      return 'minute'

def dateFromDurationDigit(base_date, duration, year, day):
    values = []
    if day is None:
        # Year
        year = int(year)
        current_year = base_date.year
        if duration == 'since':
            if year > current_year:
                year = current_year
            for i in range(year, current_year + 1):
                values.append(datetime(i, 1, 1))
            return values
        return datetime(year, base_date.month, base_date.day)

    if year is None:
        # Day
        day = int(day)
        if duration == 'since':
            return datetime(base_date.year, base_date.month, day)
        if duration == 'last':
            return datetime(base_date.year, base_date.month, day)
        return datetime(base_date.year, base_date.month, day)
    return values

# Find dates from duration
# Eg: 20 days from now
# Doesnt support 20 days from last monday
def dateFromDuration(base_date, numberAsString, unit, duration, base_time = None):
    # Check if query is `2 days before yesterday` or `day before yesterday`
    if base_time != None:
        base_date = dateFromAdverb(base_date, base_time)
    num = convert_string_to_number(numberAsString)
    if unit in day_variations:
        args = {'days': num}
    elif unit in minute_variations:
        args = {'minutes': num}
    elif unit in week_variations:
        args = {'weeks': num}
    elif unit in month_variations:
        args = {'days': 365 * num / 12}
    elif unit in year_variations:
        args = {'years': num}
    if duration == 'ago' or duration == 'before' or duration == 'earlier':
        if ('years' in args):
            return datetime(base_date.year - args['years'], base_date.month, base_date.day)
        return base_date - timedelta(**args)
    elif duration == 'after' or duration == 'later' or duration == 'from now':
        if ('years' in args):
            return datetime(base_date.year + args['years'], base_date.month, base_date.day)
        return base_date + timedelta(**args)

# Finds coming weekday
def this_week_day(base_date, weekday):
    day_of_week = base_date.weekday()
    # If today is Tuesday and the query is `this monday`
    # We should output the next_week monday
    if day_of_week > weekday:
        return next_week_day(base_date, weekday)
    start_of_this_week = base_date - timedelta(days=day_of_week + 1)
    day = start_of_this_week + timedelta(days=1)
    while day.weekday() != weekday:
        day = day + timedelta(days=1)
    return day

# Finds coming weekday
def previous_week_day(base_date, weekday):
    day = base_date - timedelta(days=1)
    while day.weekday() != weekday:
        day = day - timedelta(days=1)
    return day

def next_week_day(base_date, weekday):
    day_of_week = base_date.weekday()
    end_of_this_week = base_date + timedelta(days=6 - day_of_week)
    day = end_of_this_week + timedelta(days=1)
    while day.weekday() != weekday:
        day = day + timedelta(days=1)
    return day


# Mapping of Month name and Value
hashmonths = {
    'january': 1,
    'jan': 1,
    'february': 2,
    'feb': 2,
    'march': 3,
    'mar': 3,
    'april': 4,
    'apr': 4,
    'may': 5,
    'june': 6,
    'jun': 6,
    'july': 7,
    'jul': 7,
    'august': 8,
    'aug': 8,
    'september': 9,
    'sep': 9,
    'october': 10,
    'oct': 10,
    'november': 11,
    'nov': 11,
    'december': 12,
    'dec': 12
}

# Days to number mapping
hashweekdays = {
    'monday': 0,
    'mon': 0,
    'tuesday': 1,
    'tue': 1,
    'wednesday': 2,
    'wed': 2,
    'thursday': 3,
    'thu': 3,
    'friday': 4,
    'fri': 4,
    'saturday': 5,
    'sat': 5,
    'sunday': 6,
    'sun': 6
}

# Ordinal to number
hashordinals = {
    'first': 1,
    'one': 1,
    'second' : 2,
    'two' : 2,
    'third': 3,
    'three': 3,
    'fourth': 4,
    'four': 4,
    'forth': 4,
    'five': 5,
    'fifth': 5,
    'six': 6,
    'sixth': 6,
    'seven': 7,
    'seventh': 7,
    'eight': 8,
    'eighth': 8,
    'nine': 9,
    'ninth': 9,
    'ten': 10,
    'tenth': 10,
    'seventeen': 17,
    'last': -1
}

# Parses date
def extract_datetime(text, base_date = datetime.now):
    # Evaluate base date
    base_date = base_date()
    matches = []
    found_array = []

    # Lowercase text for easy Matching
    text = text.lower()

    # Find the position in the string
    for r, fn in regex:
        for m in r.finditer(text):
            try:
                val = fn(m, base_date)
                unit = 'day'
                if (isinstance(val, tuple)):
                  value, unit = val
                else:
                  value = val
                matches.append((m.group(), value, m.span(), unit))
            except ValueError:
                continue

    # print (matches)

    # Wrap the matched text with TAG element to prevent nested selections
    for match, value, spans, unit in matches:
        subn = re.subn('(?!<TAG[^>]*?>)' + match + '(?![^<]*?</TAG>)', '<TAG>' + match + '</TAG>', text)
        text = subn[0]
        isSubstituted = subn[1]
        if isSubstituted != 0:
            found_array.append((match, value, spans, unit))

    # To preserve order of the match, sort based on the start position
    return sorted(found_array, key = lambda match: match and match[2][0])


class DateTimeExtractor(Extractor):

    def __init__(self, base_time=datetime.now):
        self.base_time = base_time

    def __call__(self, x):
        # returns a list of tuples
        # each tuple is of the form
        # ((<start index>, <end index>), <DateTime Object>)
        dates = extract_datetime(x, self.base_time)
        y = []
        for d in dates:
            y.append((d[2], DateTime(d[1], d[0])))
        return y
