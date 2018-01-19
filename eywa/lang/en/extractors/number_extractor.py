# -*- coding: utf-8 -*-
from ....entities import Number
from .extractor import Extractor
import re

hash_units = {
  "zero": 0,
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
  "eleven": 11,
  "twelve": 12,
  "thirteen": 13,
  "fourteen": 14,
  "fifteen": 15,
  "sixteen": 16,
  "seventeen": 17,
  "eighteen": 18,
  "nineteen": 19
}

hash_tens = {
  'twenty': 20,
  'thirty': 30,
  'forty': 40,
  'fourty': 40,
  'fifty': 50,
  'sixty': 60,
  'seventy': 70,
  'eighty': 80,
  'ninety': 90
}

hash_scales = {
  "hundred": 100,
  "thousand": 1000,
  "million": 1000000
}

re_postfix = 'nd|th|rd'
re_mods = 'over|under|below|above'
re_units = '|'.join(hash_units.keys())
re_tens = '|'.join(hash_tens.keys())
re_scales = '|'.join(hash_scales.keys())

regex = [
    # Captures scales
    # nine hundred
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<units>%s)
          (\s)
          (?P<scales>%s)
        )
        (\b)
        '''%(re_mods, re_units, re_scales),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_units_scales(m.group('units'), m.group('scales'), m.group('mods'))
    ),
    # Captures
    # Ninety nine, fourty eight
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<tens>%s)
          (\s|\-)
          (?P<units>%s)
        )
        (\b)
        '''%(re_mods, re_tens, re_units),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_hash_tens(m.group('tens'), m.group('units'), m.group('mods'))
    ),
    # Captures
    # Ninety
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<tens>%s)
        )
        (\b)
        '''%(re_mods, re_tens),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_hash_tens(m.group('tens'), None, m.group('mods'))
    ),
    # Captures
    # one, two
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<units>%s)
        )
        (\b)
        '''%(re_mods, re_units),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_hash_tens(None, m.group('units'), m.group('mods'))
    ),
    # Captures
    # hundred
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<scales>%s)
        )
        (\b)
        '''%(re_mods, re_scales),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_units_scales(None, m.group('scales'), m.group('mods'))
    ),
    # Captures ranges
    # 45-60
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<digits_start>\d*\.?\d+?)
          (\s)?
          (-|to)
          (\s)?
          (?P<digits_end>\d*\.?\d+?)
        )
        (\b)
        '''%(re_mods),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_to_range(m.group('digits_start'), m.group('digits_end'))
    ),
    # Captures all digits
    # 45
    (re.compile(
        r'''
        (\b)
        (
          (?P<mods>(%s)\s)?
          (?P<digits>\d*\.?\d+?)
          (%s)?
        )
        (\b)
        '''%(re_mods, re_postfix),
        (re.VERBOSE | re.IGNORECASE)
        ),
        lambda m: convert_to_pure_number(m.group('digits'), m.group('mods'))
    ),
]

def convert_hash_tens (tens = None, units = None, mods = None):
  value = (
    hash_tens[tens] if tens is not None else 0
  )  + (
    hash_units[units] if units is not None else 0
  )

  if mods is not None:
    value += get_mods_value(mods)
  return value

def convert_units_scales (units = None, scales = None, mods = None):
  value =  (
    hash_units[units] if units is not None else 1
  )  * (
    hash_scales[scales] if scales is not None else 0
  )

  if mods is not None:
    value += get_mods_value(mods)
  return value

def convert_to_pure_number (num, mods = None):
  value = normalize_int_float(float(num))

  if mods is not None:
    value += get_mods_value(mods)
  return value

def normalize_int_float (value):
  if value.is_integer():
    value = int(value)
  return value

def convert_to_range (start, end):
  start = normalize_int_float(float(start))
  end = normalize_int_float(float(end))
  return [start, end]


def get_mods_value (mod):
  mod = mod.strip()
  if mod == 'above' or mod == 'over' or mod == 'more than':
    return 1
  if mod == 'below' or mod == 'under' or mod == 'less than':
    return -1

# Parses date
def extract_numbers(text):
    matches = []
    found_array = []

    # Lowercase text for easy Matching
    text = text.lower()

    unit = 'number'

    # Find the position in the string
    for r, fn in regex:
        for m in r.finditer(text):
            matches.append((m.group(), fn(m), m.span(), unit))


    # Wrap the matched text with TAG element to prevent nested selections
    for match, value, spans, unit in matches:
        subn = re.subn('(?!<NUMBER_TAG[^>]*?>)' + match + '(?![^<]*?</NUMBER_TAG>)', '<NUMBER_TAG>' + match + '</NUMBER_TAG>', text)
        text = subn[0]
        isSubstituted = subn[1]
        if isSubstituted != 0:
            found_array.append((match, value, spans, unit))

    # To preserve order of the match, sort based on the start position
    return sorted(found_array, key = lambda match: match and match[2][0])


class NumberExtractor(Extractor):

  def __call__(self, x):
    return [(y[2], Number(y[1], y[0])) for y in extract_numbers(x)]
