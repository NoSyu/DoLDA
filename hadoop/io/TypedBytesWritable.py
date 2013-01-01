'''
JinYeong Bak(jy.bak@kaist.ac.kr)
TypedBytesWritable

We get source code from https://github.com/klbostee/typedbytes and modified it

'''

from Writable import AbstractValueWritable

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

UNICODE_ENCODING = 'utf8'

unpack_type = Struct('!B').unpack
unpack_byte = Struct('!b').unpack
unpack_int = Struct('!i').unpack
unpack_long = Struct('!q').unpack
unpack_float = Struct('!f').unpack
unpack_double = Struct('!d').unpack

_len = len

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

class Bytes(str):
    def __repr__(self):
        return "Bytes(" + str.__repr__(self) + ")"

class TypedBytesWritable(AbstractValueWritable):
    def __init__(self, file, unicode_errors='strict'):
        self.file = None
        self.unicode_errors = unicode_errors
        self.eof = False
        self.handler_table = self._make_handler_table()
    
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
            
    def write(self, data_output):
        #data_output.writeInt(self._value)
        pass
        
    def readFields(self, data_input):
        try:
            self.file = data_input
            t = unpack_type(data_input.read(1))[0]
            self.t = t
        except StructError:
            self.eof = True
            raise StopIteration
        self._value = self.handler_table[t](self)
    
    def invalid_typecode(self):
        raise StructError("Invalid type byte: " + str(self.t))
    
    def _make_handler_table(self):
        return list(TYPECODE_HANDLER_MAP.get(i,
                    self.invalid_typecode) for i in xrange(256))
                    
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