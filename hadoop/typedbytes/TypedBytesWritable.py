'''
JinYeong Bak(jy.bak@kaist.ac.kr)
TypedBytesWritable

It's based on https://github.com/klbostee/typedbytes
'''

from hadoop.io import *

from cPickle import dumps, loads, UnpicklingError, HIGHEST_PROTOCOL
from struct import pack, unpack, error as StructError
from array import array
try:
    from struct import Struct
except ImportError:
    class Struct(object):
        def __init__(self, fmt):
            self.fmt = fmt
        def unpack(self, *args):
            return unpack(self.fmt, *args)
        def pack(self, *args):
            return pack(self.fmt, *args)

from types import BooleanType, IntType, LongType, FloatType 
from types import UnicodeType, StringType, TupleType, ListType, DictType
from datetime import datetime, date
from decimal import Decimal

# Typed bytes types:
BYTES = 0
BYTE = 1
BOOL = 2
INT = 3
LONG = 4
FLOAT = 5
DOUBLE = 6
STRING = 7
VECTOR = 8
LIST = 9
MAP = 10

# Application-specific types:
PICKLE = 100
BYTESTRING = 101

# Low-level types:
MARKER = 255

UNICODE_ENCODING = 'utf8'

unpack_type = Struct('!B').unpack
unpack_byte = Struct('!b').unpack
unpack_int = Struct('!i').unpack
unpack_long = Struct('!q').unpack
unpack_float = Struct('!f').unpack
unpack_double = Struct('!d').unpack

_len = len


_BYTES, _BYTE, _BOOL = BYTES, BYTE, BOOL
_INT, _LONG, _FLOAT, _DOUBLE = INT, LONG, FLOAT, DOUBLE
_STRING, _VECTOR, _LIST, _MAP = STRING, VECTOR, LIST, MAP
_PICKLE, _BYTESTRING, _MARKER = PICKLE, BYTESTRING, MARKER

LIST_CODE, MARKER_CODE = (pack('!B', i) for i in (LIST, MARKER))

pack_len = Struct('!i').pack
pack_byte = Struct('!Bb').pack
pack_int = Struct('!Bi').pack
#pack_len_type = Struct('!iB').pack
pack_long = Struct('!Bq').pack
pack_float = Struct('!Bf').pack
pack_double = Struct('!Bd').pack

_int, _type = int, type


def flatten(iterable):
    for i in iterable:
        for j in i:
            yield j

class Bytes(str):
    def __repr__(self):
        return "Bytes(" + str.__repr__(self) + ")"

class TypedBytesWritable(AbstractValueWritable):
    def __init__(self, unicode_errors='strict'):
        self.file = None
        self.unicode_errors = unicode_errors
        self.eof = False
        self.handler_table = self._make_handler_table()
        self._value = None
        self.handler_map = self._make_handler_map()
        
    '''
    Default procedure
    '''
    def write(self, data_output):
        #data_output.writeInt(self._value)
        try:
            self.out_file = data_output
            #print _type(self._value)
            writefunc = self.handler_map[_type(self._value)]           
        except KeyError:
            writefunc = self.write_pickle
        write_one = writefunc(self, self._value)
        
    
    def readFields(self, data_input):
        try:
            self.file = data_input
            key_length = data_input.readInt()
            key_type = data_input.readByte()
            self.t = key_type
        except StructError:
            self.eof = True
            raise StopIteration
        #print key_type
        self._value = self.handler_table[key_type](self)
        
        
    '''
    Reader
    ''' 
    def _read(self):
        try:
            t = unpack_type(self.file.read(1))[0]
            self.t = t
        except StructError:
            self.eof = True
            raise StopIteration
        return self.handler_table[t](self)
    
    def _reads(self):
        r = self._read
        while 1:
            yield r()
            
    def read_bytes(self):
        count = unpack_int(self.file.read(4))[0]
        value = self.file.read(count)
        if _len(value) != count:
            raise StructError("EOF before reading all of bytes type")
        return Bytes(value)

    def read_byte(self):
        return unpack_byte(self.file.read(1))[0]

    def read_bool(self):
        return bool(unpack_byte(self.file.read(1))[0])

    def read_int(self):
        return unpack_int(self.file.read(4))[0]

    def read_long(self):
        return unpack_long(self.file.read(8))[0]

    def read_float(self):
        return unpack_float(self.file.read(4))[0]

    def read_double(self):
        return unpack_double(self.file.read(8))[0]

    def read_string(self):
        count = unpack_int(self.file.read(4))[0]
        value = self.file.read(count)
        if _len(value) != count:
            raise StructError("EOF before reading all of string")
        return value

    read_bytestring = read_string

    def read_unicode(self):
        count = unpack_int(self.file.read(4))[0]
        value = self.file.read(count)
        if _len(value) != count:
            raise StructError("EOF before reading all of string")
        return value.decode(UNICODE_ENCODING, self.unicode_errors)

    def read_vector(self):
        r = self._read
        count = unpack_int(self.file.read(4))[0]
        try:
            return tuple(r() for i in xrange(count))
        except StopIteration:
            raise StructError("EOF before all vector elements read")

    def read_list(self):
        value = list(self._reads())
        if self.eof:
            raise StructError("EOF before end-of-list marker")
        return value

    def read_map(self):
        r = self._read
        count = unpack_int(self.file.read(4))[0]
        return dict((r(), r()) for i in xrange(count))

    def read_pickle(self):
        count = unpack_int(self.file.read(4))[0]
        bytes = self.file.read(count)
        if _len(bytes) != count:
            raise StructError("EOF before reading all of bytes type")
        return loads(bytes)

    def read_marker(self):
        raise StopIteration
        
    TYPECODE_HANDLER_MAP = {
        BYTES: read_bytes,
        BYTE: read_byte,
        BOOL: read_bool,
        INT: read_int,
        LONG: read_long,
        FLOAT: read_float,
        DOUBLE: read_double,
        STRING: read_string,
        VECTOR: read_vector,
        LIST: read_list,
        MAP: read_map,
        PICKLE: read_pickle,
        BYTESTRING: read_bytestring,
        MARKER: read_marker
    }
    
    def invalid_typecode(self):
        raise StructError("Invalid type byte: " + str(self.t))
    
    def _make_handler_table(self):
        return list(self.TYPECODE_HANDLER_MAP.get(i,
                    self.invalid_typecode) for i in xrange(256))

    '''
    Writer
    Modified only write_strings
    Should write all length
    '''
    def write_bytes(self, bytes):
        self.out_file.write(pack_int(_BYTES, _len(bytes)))
        self.out_file.write(bytes)

    def write_byte(self, byte):
        self.out_file.write(pack_byte(_BYTE, byte))

    def write_bool(self, bool_):
        self.out_file.write(pack_byte(_BOOL, _int(bool_)))

    def write_int(self, int_):
        # Python ints are 64-bit
        if -2147483648 <= int_ <= 2147483647:
            self.out_file.write(pack_int(_INT, int_))
        else:
            self.out_file.write(pack_long(_LONG, int_))

    def write_long(self, long_):
        # Python longs are infinite precision
        if -9223372036854775808L <= long_ <= 9223372036854775807L:
            self.out_file.write(pack_long(_LONG, long_))
        else:
            self.write_pickle(long_)

    def write_float(self, float_):
        self.out_file.write(pack_float(_FLOAT, float_))

    def write_double(self, double):
        self.out_file.write(pack_double(_DOUBLE, double))

    def write_string(self, string):
        all_len = 1 + 4 + _len(string)
        self.out_file.write(pack_len(all_len))
        self.out_file.write(pack_int(_STRING, _len(string)))
        #self.out_file.write(pack_len_type(_len(string), _STRING))
        self.out_file.write(string)

    def write_bytestring(self, string):
        self.out_file.write(pack_int(_BYTESTRING, _len(string)))
        self.out_file.write(string)

    def write_unicode(self, string):
        string = string.encode(UNICODE_ENCODING, self.unicode_errors)
        self.out_file.write(pack_int(_STRING, _len(string)))
        self.out_file.write(string)

    def write_vector(self, vector):
        self.out_file.write(pack_int(_VECTOR, _len(vector)))
        self._writes(vector)

    def write_list(self, list_):
        self.out_file.write(LIST_CODE)
        self._writes(list_)
        self.out_file.write(MARKER_CODE)

    def write_map(self, map):
        self.out_file.write(pack_int(_MAP, _len(map)))
        self._writes(flatten(map.iteritems()))

    def write_pickle(self, obj):
        bytes = dumps(obj, HIGHEST_PROTOCOL)
        self.out_file.write(pack_int(_PICKLE, _len(bytes)))
        self.out_file.write(bytes)

    def write_array(self, arr):
        bytes = arr.tostring()
        self.out_file.write(pack_int(_BYTES, _len(bytes)))
        self.out_file.write(bytes)

    TYPE_HANDLER_MAP = {
        BooleanType: write_bool,
        IntType: write_int,                
        LongType: write_long,       
        FloatType: write_double,
        StringType: write_string,
        TupleType: write_vector,
        ListType: write_list,        
        DictType: write_map,
        UnicodeType: write_unicode,
        Bytes: write_bytes,
        datetime: write_pickle,
        date: write_pickle,
        Decimal: write_pickle,
        array: write_array
    }

    def _make_handler_map(self):
        return dict(self.TYPE_HANDLER_MAP)