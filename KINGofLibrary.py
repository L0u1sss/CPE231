import ctypes
import sys

class Array :
    '''  A simple array class using ctypes for memory management.'''
    def __init__(self,size):
        assert size > 0, "Size must be a positive integer."
        self._size = size
        self.PyArrayType = ctypes.py_object * size
        self._elements = self.PyArrayType()
        self.clear(None)

    def __len__(self):
        return self._size
    
    def __getitem__(self,index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            length = len(range(start, stop, step)) # ความยาวชิ้นที่ดึง step ทั้งบวก/ลบ
            new_arr = Array(length)
            k = 0
            for j in range(start, stop, step):
                new_arr[k] = self._elements[j]
                k += 1
            return new_arr
        else:
            if index < 0: #index ติดลบ (-1 คือท้ายสุด)
                index += self._size
            assert 0 <= index < self._size , "Array subscript out of range."
            return self._elements[index]

    def __setitem__(self,index,value):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            if isinstance(value, Array): # แปลง source เป็น iterable ธรรมดา
                src = [value[i] for i in range(len(value))]
            else:
                src = list(value)
           
            dst_indices = list(range(start, stop, step))
            assert len(src) == len(dst_indices), \
                "Attempt to assign sequence of size {} to extended slice of size {}".format(len(src), len(dst_indices))
            for j, v in zip(dst_indices, src):
                self._elements[j] = v
        else:
            if index < 0:
                index += self._size
        assert 0 <= index < self._size , "Array subscript out of range."
        self._elements[index] = value

    def clear(self,value):
        '''Clear the array by setting all elements to the specified value.'''
        for i in range(self._size):
            self._elements[i] = value

    def copy(self):
        new_array = Array(self._size)
        for i in range(self._size):
            new_array[i] = self._elements[i]
        return new_array

    def __repr__(self):
        s = '['
        for x in self._elements:
            s = s + str(x) + ', '
        s = s[:-2] + ']' 
        return s

    def __iter__(self):
        self._curIndex = 0
        return self
    
    def __next__(self):
        if self._curIndex < len(self._elements):
            entry = self._elements[self._curIndex]
            self._curIndex += 1
            return entry
        else:
            raise StopIteration

class Array2D:
    '''This is 2D Array class'''
    def __init__(self,numRows,numCols):
        self._theRows = Array(numRows)
        for i in range(numRows):
            self._theRows[i] = Array(numCols)

    def numRows(self):
        return len(self._theRows)
    
    def numCols(self):
        return len(self._theRows[0])
    
    def clear(self,value):
        for row in self._theRows:
            row.clear(value)

    def __getitem__(self,indexTuple):
        assert len(indexTuple) == 2, "Invalid number of array subscripts."
        row = indexTuple[0]
        col = indexTuple[1]
        assert row >= 0 and row < self.numRows() and col >= 0 and col < self.numCols() ,"Array subscript out of range"
        return self._theRows[row][col]

    def __setitem__(self,indexTuple,value):
        assert len(indexTuple) == 2, "Invalid number of array subscripts."
        row = indexTuple[0]
        col = indexTuple[1]
        assert row >= 0 and row < self.numRows() and col >= 0 and col < self.numCols(), "Array subscript out of range"
        self._theRows[row][col] = value

    def __repr__(self):
        s = ""
        for i in self._theRows:
            s += str(i) + "\n"
        return s[:-1]

class Matrix(Array2D):
    '''This is Matrix'''
    def __init__(self, numRows, numCols):
        super().__init__(numRows, numCols)
        super().clear(0) #สร้าง newmetrix ใหม่ให้เป็น 0 ให้หมด

    def scaleBy(self,scalar):
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                self[r, c] *= scalar
        return self

    def __add__(self , rhsMatrix):
        assert rhsMatrix.numRows() == self.numRows() and rhsMatrix.numCols() == self.numCols() , "Matrix sizes not compatible for the add operation"
        newMatrix = Matrix (self.numRows(),self.numCols())
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                newMatrix[ r,c ] = self[ r,c ] + rhsMatrix[ r,c ]
        return newMatrix

    def __sub__(self , rhsMatrix):
        assert rhsMatrix.numRows() == self.numRows() and rhsMatrix.numCols() == self.numCols() , "Matrix sizes not compatible for the add operation"
        newMatrix = Matrix (self.numRows(),self.numCols())
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                newMatrix[ r,c ] = self[ r,c ] - rhsMatrix[ r,c ]
        return newMatrix
    
    def transpose(self):
        newMatrix = Matrix(self.numCols(),self.numRows())
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                newMatrix[c,r] = self[r,c]
        return newMatrix
        
    def __mul__(self , rhsMatrix):
        assert (self.numCols() == rhsMatrix.numRows()) , "Matrix sizes not compatible for the add operation"
        newMatrix = Matrix (self.numRows(),rhsMatrix.numCols())
        for r in range(newMatrix.numRows()):
            for c in range(newMatrix.numCols()):            
                for i in range(self.numCols()):
                    newMatrix[r,c] += self[r,i] * rhsMatrix[i,c]
        return newMatrix

#def is_square(self):
     #   """ Check if the matrix is square """
      #  return self.rows == self.cols

    #def determinant(self, mat=None):
        """ Recursively calculate the determinant of a matrix """
        if mat is None:
            mat = self.matrix
        
        # Base case for 2x2 matrix
        if len(mat) == 2:
            return mat[0,0] * mat[1,1] - mat[0,1] * mat[1,0]

        det = 0
        for c in range(len(mat)):
            det += ((-1) ** c) * mat[0][c] * self.determinant(self.get_minor(mat, 0, c))
        return det

    #def get_minor(self, mat, row, col):
        """ Get the minor matrix after removing the specified row and column """
        return [row[:col] + row[col+1:] for row in (mat[:row] + mat[row+1:])]


    #def cofactor(self):
        """ Calculate the cofactor matrix """
        cofactor_matrix = []
        for r in range(self.rows):
            cofactor_row = []
            for c in range(self.cols):
                minor = self.get_minor(self.matrix, r, c)
                cofactor_row.append(((-1) ** (r + c)) * self.determinant(minor))
            cofactor_matrix.append(cofactor_row)
        return cofactor_matrix

    #def inverse(self):
        """ Calculate the inverse of the matrix """
        if not self.is_square():
            raise ValueError("Inverse is only defined for square matrices")
        
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Transpose of the cofactor matrix divided by the determinant
        cofactor_matrix = self.cofactor()
        adjugate = Matrix(cofactor_matrix).transpose()
        return [[adjugate[i][j] / det for j in range(self.cols)] for i in range(self.rows)]


class LStack :
    '''This is Stack that implemented by Python List '''
    #Create an empty stack.
    def __init__(self):
        self._theItems = list()

    def isEmpty(self):
        return len(self) == 0 
    
    def __len__ (self):
        return len (self._theItems)
    
    def peek(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        return self._theItems[-1]
    
    def pop(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        return self._theItems.pop()
    
    def push(self, item):
        self._theItems.append(item)

    def __repr__(self):
        s = ""
        for item in reversed(self._theItems):
            s = s + "|" + str(item) + "\t|" + "\n" 
        s = s + ("___" * len(str(item[-1])))
        return s
    
class AStack:
    def __init__(self,size = 10):
        self._theItem = Array(size)
        self._top = 0
        self._capacity = size

    def isEmpty(self):
        return len(self) == 0 
    
    def __len__ (self):
        return self._top #กำหนดการเช็ค len ของ stack 
    
    def peek(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        return self._theItems[self._top - 1]
    
    def pop(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        item = self._theItem[self._top-1]
        self._theItem[self._top-1] = None
        self._top -= 1
        return item
    
    def push(self, item):
        assert self._top < self._capacity, "Stack is overflow"
        self._theItem[self._top] = item
        self._top += 1

    def __repr__(self):
        s = ""
        for item in reversed(self._theItem):
            s = s + "|" + str(item) + "\t|" + "\n" 
        s = s + ("_____" * len(str(item[-1])))
        return s
    
class LQueue:
    #Create ab enoty queue.
    def __init__(self):
        self._qList = list()
    
    def isEmpty(self):
        return len(self) == 0
    
    def __len__(self):
        return len(self._qList)
    
    def enqueue(self,item):
        self._qList.append(item)
    
    def dequeue(self):
        assert not self.isEmpty() , "Cannot dequeue from an empty queue."
        return self._qList.pop(0)
    
    def __repr__(self):
        s = "<--"
        s = s.join([str(x) for x in self._qList])
        return s    
    
class AQueue:
    def __init__(self,size = 10):
        self._theItems = Array(size)
        self._capacity = size
        self._back = 0

    def isEmpty(self):
        return len(self) == 0
    
    def __len__(self):
        return self._back
    
    def enqueue(self,item):
        assert self._back < self._capacity , "Queue is already full" #เช็คก่อนว่าเต็มรึยัง 
        self._theItems[self._back] = item #ใส่ตรงตำแหน่งที่ back ชี้อยู่
        self._back += 1
    
    def dequeue(self):
        assert not self.isEmpty() , "Cannot dequeue from an empty queue."
        item = self._theItems[0]
        for i in range(self._back-1): #ตั้งแต่ i ตัวแรกถึง i สุดท้าย
            self._theItems[i] = self._theItems[i+1]
        self._theItems[self._back-1] = None
        self._back -= 1
        return item
    
    def __repr__(self):
        s = "<--"
        s = s.join([str(x) for x in self._theItems])
        return s
    
class CQueue:
    def __init__(self,size = 10):
        self._theItem = Array(size)   # ใช้ I ใหญ่ให้ตรงกัน
        self._capacity = size
        self._front = 0
        self._back = 0
        self._count = 0
    
    def isEmpty(self):   # แก้สะกดชื่อ
        return self._count == 0
    
    def isFull(self):
        return self._count == self._capacity
    
    def enqueue(self, item):
        assert not self.isFull(), "Queue is already full"
        self._theItem[self._back] = item
        self._back = (self._back + 1) % self._capacity
        self._count += 1

    def dequeue(self):
        assert not self.isEmpty(), "Cannot dequeue from an empty queue."
        item = self._theItem[self._front]
        self._theItem[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._count -= 1
        return item
    
    def length(self):
        return self._count
    
    def __repr__(self):
        if self.isEmpty():
            return "Queue is empty"
        result = []
        i = self._front
        for _ in range(self._count):
            result.append(str(self._theItem[i]))
            i = (i + 1) % self._capacity
        return " <- ".join(result)
    
class Deque():
    def __init__(self,size = 10):
        self._items = Array(size)
        self._front = 0
        self._back = 0 #มันก็คือ rear นั่นแหล่ะ 
        self._count = 0
        self._capacity = size

    def isEmpty(self):
        return self._count == 0
    
    def isFull(self):
        return self._count == self._capacity
    
    def length(self):
        return self._count
    
    ##
    def addRear(self, item):
        assert not self.isFull(), "Deque is full"
        self._items[self._back] = item
        self._back = (self._back + 1) % self._capacity  # ใช้ modulo ให้ index หมุนวน
        self._count += 1

    ##
    def addFirst(self, item):
        # เลื่อนค่าทุกตัวไปขวา 1 ตำแหน่ง
        for i in range(self._count, 0, -1):
            self._items[i] = self._items[i - 1]
        # ใส่ค่าใหม่ที่ index 0
        self._back += 1
        self._items[0] = item
        self._count += 1


    ##
    def deleteRear(self):
        assert not self.isEmpty(), "Deque is empty"
        self._back = (self._back - 1 + self._capacity) % self._capacity
        item = self._items[self._back]
        self._items[self._back] = None
        self._count -= 1
        return item
    

    
    def deleteFirst(self):
        assert not self.isEmpty(), "Deque is empty"
        item = self._items[0]
        for i in range(0, self._count,1):
            self._items[i] = self._items[i + 1]
        self._items[self._count] = None
        self._count -= 1
        return item
    
    ##
    def first(self):
        assert not self.isEmpty(), "Deque is empty"
        item = self._items[self._front]
        return item

    ##
    def rear(self):
        assert not self.isEmpty(), "Deque is empty"
        item = self._items[self._count-1]
        return item

    ##
    def __repr__(self):
        s = "-> "
        i = self._front
        for _ in range(self._count):
            s += "[" + str(self._items[i]) + "]"
            i = (i + 1) % self._capacity
        s += " <-"
        return s
    
class _SLinkNode:  #ตัวคอนเทนเนอร์เก็บข้อมูล
    '''This is just a node of linked list'''
    def __init__(self,item):
        self._item = item
        self._next = None
        
class SLinkedList:
    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0
    
    def __len__(self):
        return self._size
    
    def isEmpty(self):
        return len(self) == 0

    def prepend(self,item):
        newNode = _SLinkNode(item)
        if self.isEmpty():
            self._tail = newNode
        else:
            newNode._next = self._head
        self._head = newNode
        self._size += 1

    def append(self,item):
        newNode = _SLinkNode(item)
        if self.isEmpty():
            self._head = newNode
        else:
            self._tail._next = newNode
        self._tail = newNode
        self._size += 1

    def __repr__(self):
        curNode = self._head
        s = "["
        while curNode is not None:
            s = s + str(curNode._item) + "->"
            curNode = curNode._next

        s = s[:-2] + "]"
        return s 
    
    def __contains__(self,target):
        '''use with operation "in" '''
        curNode = self._head
        while curNode is not None and curNode._item != target:
            curNode = curNode._next
        return curNode is not None
    
    def remove(self,item):
        predNode = None
        curNode = self._head
        while curNode is not None and curNode._item != item:
            predNode = curNode
            curNode = curNode._next
        assert curNode is not None, "The item must be in this link"
        self._size -= 1
        if curNode is self._head:
            self._head = curNode._next
        elif curNode is self._tail:
            self._tail = predNode
            self._tail._next = None
        else:
            predNode._next = curNode._next
        return curNode._item
    
class LLStack(SLinkedList):
    """This LLStack is inherited from SlinkedList and top is head"""

    def push(self,item):
        self.prepend(item)
    
    def pop(self):
        assert not self.isEmpty(), "Cannot pop from an empty stack"
        curNode = self._head
        item = curNode._item
        self._head = curNode._next
        self._size -=1
        return item
    
    def peek(self):
        assert not self.isEmpty(), "Cannot pop from an empty stack"
        return self._head._item
    
    def __repr__(self):
        curNode = self._head
        s = "-------------------\n"
        while curNode is not None:
            s = s+str(curNode._item) + "\n"
            curNode = curNode._next
        s = s + "---------------------" 
        return s
    
class LLQueue(SLinkedList):
    """This LLQueue is inherited from SlinkedList"""
    def __init__(self):
        super().__init__()
        
    def enqueue(self,item):
        self.append(item)

    def dequeue(self):
        assert not self.isEmpty() , "Cannot dequeue from an empty queue."
        curNode = self._head
        item = curNode._item
        self._size -= 1
        self._head = curNode._next

        return item

    def __repr__(self):
        return super().__repr__()
    
class _DLinkNode(object):
    def __init__(self,item,prev,next):
        self._item = item
        self._prev = prev
        self._next = next
    
class DLinkedlist:
    """This is doubly linked list"""
    def __init__(self):
        self._header = _DLinkNode(None,None,None)
        self._trailer = _DLinkNode(None,None,None)
        self._header._next = self._trailer
        self._trailer._prev = self._header
        self._size = 0
    
    def insert_between(self,item,predecessor,successor):
        newNode = _DLinkNode(item,predecessor,successor)
        predecessor._next = newNode
        successor._prev = newNode
        self._size += 1

    def delete_node(self,node):
        assert not self.isEmpty() , "Doubly Linked list is empty"
        predecessor = node._prev
        successor = node._next
        predecessor._next = successor
        successor._prev = predecessor
        self._size -= 1
        item =  node._item
        node._prev = node._next = node._tem = None
        return item

    def isEmpty(self):
        return self._size == 0
    
    def __len__(self):
        return self._size
    
    def __repr__(self):
        curNode = self._header 
        s = "["
        while curNode is not None:
            s = s + str(curNode._item) + "<->"
            curNode = curNode._next
        s = s[:-3] + "]"
        return s
    
class DLDeque(DLinkedlist):
    def __init__(self):
        super().__init__()

    def first(self):
        assert not self.isEmpty(), "Deque is empty"
        return self._header._next._item
    
    def rear(self):
        assert not self.isEmpty(), "Deque is empty"
        return self._trailer._prev._item
    
    def addFirst(self,item):
        self.insert_between(item, self._header, self._header._next)

    def addRear(self,item):
        self.insert_between(item, self._trailer._prev, self._trailer)

    def deleteFirst(self):
        assert not self.isEmpty(), "Deque is empty"
        self.delete_node(self._header._next)

    def deleteRear(self):
        assert not self.isEmpty(), "Deque is empty"
        self.delete_node(self._trailer._prev)

    def __repr__(self):
        cur = self._header._next
        s = "["
        while cur is not self._trailer:
            s += f"{cur._item}<->"
            cur = cur._next
        return s[:-3] + "]" if self._size > 0 else "[]"
    
def bubbleSort(seq):
    n = len(seq) - 1
    for i in range(n,0,-1):
        for j in range(i):
            if seq[j] > seq[j+1]:
                seq[j], seq[j+1] = seq[j+1], seq[j]
                #print(seq)
    return seq

def selectionSort(seq):
    n = len(seq)
    for i in range(n-1):
        smallNdx = i
        for j in range(i+1, n):
            if seq[j] < seq[smallNdx]:
                smallNdx = j
        if smallNdx != i:
            seq[i], seq[smallNdx] = seq[smallNdx], seq[i]

def insertionSort(seq):
    n = len(seq)
    for i in range(1, n):
        value = seq[i]
        pos = i
        while pos > 0 and value < seq[pos - 1]:
            seq[pos] = seq[pos - 1]
            pos -= 1
        seq[pos] = value
    return seq

def insertionSort_reverse(seq):
    n = len(seq)
    for i in range(1, n):
        value = seq[i]
        pos = i
        while pos > 0 and value > seq[pos - 1]:
            seq[pos] = seq[pos - 1]
            pos -= 1
        seq[pos] = value
    return seq

def merge(left,right,seq):
    i = j = 0
    while i + j < len(seq):
        if j == len(right) or (i < len(left) and left[i] < right[j]):
            seq[i + j] = left[i]
            i += 1
        else:
            seq[i + j] = right[j]
            j += 1
    return seq

def mergeSort(seq):
    n = len(seq)
    if n < 2:
        return
    mid = n // 2
    left = seq[0:mid]
    right = seq[mid:n]
    mergeSort(left)
    mergeSort(right)
    
    seq = merge(left,right,seq)
    return seq

def quickSort(seq):
    sys.setrecursionlimit(10000)
    if len(seq) < 2 :return seq
    pi = seq[0]
    seq = seq[1:]
    lo = [x for x in seq if x < pi]
    hi = [x for x in seq if x >= pi]
    return quickSort(lo) + [pi] + quickSort(hi)

def quickSort_reverse(seq):
    import sys
    sys.setrecursionlimit(10000)
    if len(seq) < 2:
        return seq
    pi = seq[0]
    seq = seq[1:]
    lo = [x for x in seq if x > pi]      
    hi = [x for x in seq if x <= pi]    
    return quickSort_reverse(lo) + [pi] + quickSort_reverse(hi)

def radixSort(seq, numDigits):
    column = 1
    for _ in range(numDigits):
        bins = [[] for _ in range(10)]     
        for x in seq:
            d = (x // column) % 10
            bins[d].append(x)
        i = 0
        for b in range(10):
            for v in bins[b]:           
                seq[i] = v
                i += 1
        column *= 10
    return seq

class Item:
    __slots__ = '_key','_value'
    def __init__(self,k,v):
        self._key = k
        self._value = v
        
    def __eq__(self,other):
        return self._key == other._key
    
    def __ne__(self,other):
        return not(self == other)
    
    def __lt__(self,other):
        return self._key < other._key
        
    def __repr__(self):
        return str(self._key) + ':' + str(self._value)
        
    def __str__(self):
        return str(self._key) + ':' + str(self._value)
    
from collections.abc import MutableMapping
class LinearMap(MutableMapping):
    
    def __init__(self):
        self._table = []
        
    def __getitem__(self,k):
        for item in self._table:
            if k == item._key:
                return item._value
        raise KeyError('Key Error: '+ repr(k))
        
    def __setitem__(self,k,v):
        for item in self._table:
            if k == item._key:
                item._value = v
                return
        self._table.append(Item(k,v))
    
    def __delitem__(self,k):
        for j in range(len(self._table)):
            if k == self._table[j]._key:
                self._table.pop(j)
                return
        raise KeyError('Key Error: '+ repr(k))
        
    def __len__(self):
        return len(self._table)
    
    def __iter__(self):
        for item in self._table:
            yield item._key
            
    def __repr__(self):
        if len(self) == 0:
            s = "[ ]"
        else:
            s = '['
            for item in self._table:
                s += str(item)
                s += ', '
            s = s[:-2]
            s += ']'
        return s
    
from collections.abc import MutableMapping
class ChainHashMap(MutableMapping):
    """Hash map implemented with separate chaining for collision resolution."""
#-----------------------------------------------        
    def __init__(self,cap = 11):
        """Create an empty hash-table map."""
        self._table = cap * [None]
        self._n = 0 #for count the items in MAP not equal to No. of table
        
    def _hash_function(self,k):
        sum = 0
        for i in k:
            sum += ord(i)
        return  sum % len(self._table)
    
    def __len__(self):
        return self._n
    
    def __getitem__(self,k):
        j = self._hash_function(k) # ใช้ key มาผ่าน hashfunction 
        return self._bucket_getitem(j,k)
    
    def __setitem__(self,k,v):
        j = self._hash_function(k)
        self._bucket_setitem(j,k,v)
        if self._n > 8 * len(self._table) //10 :
            self._resize(2 * len(self._table) - 1) #เพิ่มขนาดเป็น 2 เท่า
    
    def __delitem__(self,k):
        j = self._hash_function(k)
        self._bucket_delitem(j,k)
        self._n -= 1
        
    def _resize(self,c):
        old = list(self.items())
        self._table = c * [None]
        self._n = 0
        for (k,v) in old:
            self[k] = v
 #---------------------------------------------------------                    
    def _bucket_getitem(self,j,k):
        bucket = self._table[j]
        if bucket is None:
            raise KeyError('Key Error: '+ repr(k)) # no match found
        return bucket[k] # may raise KeyError

    def _bucket_setitem(self,j,k,v):
        if self._table[j] is None:
            self._table[j] = LinearMap() # bucket is new to the table note that it use LinearMap
        oldsize = len(self._table[j])
        self._table[j][k] = v
        if len(self._table[j]) > oldsize: # key was new to the table
            self._n += 1 # increase overall map size

    def _bucket_delitem(self,j,k):
        bucket = self._table[j]
        if bucket is None:
            raise KeyError('Key Error: '+ repr(k)) # no match found
        del(bucket[k]) # may raise KeyError

#---------------------------------------------------------                    
    def __iter__(self):
        for bucket in self._table:
            if bucket is not None: # a nonempty slot
                for key in bucket:
                    yield key
                    
    def __repr__(self):
        print("load factor = {:.2f}".format(len(self)/len(self._table)))
        if len(self) == 0:
            s = "[ ]"
        else:
            s = '['
            for item in self._table:
                s += str(item)
                s += ', '
            s = s[:-2]
            s += ']'
        return s
    
from collections.abc import MutableMapping
class ProbeHashMap(MutableMapping):
    """Hash map implemented with linear probing for collision resolution."""
#-----------------------------------------------        
    def __init__(self,cap = 11,):
        """Create an empty hash-table map."""
        self._table = cap * [None]
        self._n = 0 #for count the items in MAP
        #remind that the No. of items is not equal No. of _table
    
    def _hash_function(self,k):
        sum = 0
        for i in k:
            sum += ord(i)
        return sum % len(self._table)
    
    def __len__(self):
        return self._n
    
    def __getitem__(self,k):
        j = self._hash_function(k)
        return self._bucket_getitem(j,k)
    
    def __setitem__(self,k,v):
        j = self._hash_function(k)
        self._bucket_setitem(j,k,v)
        if self._n > len(self._table) // 2:
            self._resize(2 * len(self._table) - 1)
    
    def __delitem__(self,k):
        j = self._hash_function(k)
        self._bucket_delitem(j,k)
        self._n -= 1
        
    def _resize(self,c):
        old = list(self.items())
        self._table = c * [None]
        self._n = 0
        for (k,v) in old:
            self[k] = v
#-------------------------------------------------------            
    _AVAIL = object()
    
    def _is_available(self,j):
        """Return True if index j is available in table."""
        return (self._table[j] is None) or (self._table[j] is ProbeHashMap._AVAIL)

    def _find_slot(self,j,k):
        """Search for key k in bucket at index j.
        Return (success, index) tuple, described as follows:
        If match was found, success is True and index denotes its location.
        If no match found, success is False and index denotes first available slot."""
        firstAvail = None
        while True:
            if self._is_available(j):
                if firstAvail is None:
                    firstAvail = j # mark this as first avail
                if self._table[j] is None:
                    return (False, firstAvail) # search has failed
            elif k == self._table[j]._key:
                 return (True, j) # found a match
            j = (j + 1) % len(self._table) # keep looking (cyclically)
 
    def _bucket_getitem(self, j, k):
        found, s = self._find_slot(j, k)
        if not found:
             raise KeyError( "Key Error: "+ repr(k)) # no match found
        return self._table[s]._value

    def _bucket_setitem(self, j, k, v):
        found, s = self._find_slot(j, k)
        if not found:
            self._table[s] = Item(k,v) # insert new item
            self._n += 1 # size has increased
        else:
             self._table[s]._value = v # overwrite existing

    def _bucket_delitem(self, j, k):
        found, s = self._find_slot(j, k)
        if not found:
            raise KeyError("Key Error: "+ repr(k)) # no match found
            #self._table[s] = ProbeHashMap._AVAIL # mark as vacated
        else:
            self._table[s] = None
            
    def __iter__ (self):
        for j in range(len(self._table)): # scan entire table
             if not self._is_available(j):
                yield self._table[j]._key
                
    def __repr__(self):
        print("load factor = {:.2f}".format(len(self)/len(self._table)))
        if len(self) == 0:
            s = "[ ]"
        else:
            s = '['
            for item in self._table:
                s += str(item)
                s += ', '
            s = s[:-2]
            s += ']'
        return s
    
class BinaryTree:
    def __init__(self,key):
        self.key = key
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild is None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild is None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setNodeValue(self,obj):
        self.key = obj

    def getNodeValue(self):
        return self.key

    def preorder(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()

    def postorder(self):
        if self.leftChild:
            self.leftChild.postorder()
        if self.rightChild:
            self.rightChild.postorder()
        print(self.key)
        pass

    def inorder(self):
        if self.leftChild:
            self.leftChild.inorder()
        print(self.key)
        if self.rightChild:
            self.rightChild.inorder()
        pass

    def breadthorder(self):
        q = list()
        q.append(self)
        while q:
            node = q.pop(0)
            print(node.key)
            if node.leftChild:
                q.append(node.leftChild)
            if node.rightChild:
                q.append(node.rightChild)

class MaxHeap :
    # Create a max-heap with maximum capacity of maxSize.
    def __init__( self,maxSize):
        self._elements = Array( maxSize )
        self._count = 0

    # Return the number of items in the heap.
    def __len__( self ):
        return self._count

    # Return the maximum capacity of the heap.
    def capacity( self ):
        return len( self._elements )

    # Add a new value to the heap.
    def add( self, value ):
        assert self._count < self.capacity(), "Cannot add to a full heap."
        # Add the new value to the end of the list.
        self._elements[ self._count ] = value
        self._count += 1
        # Sift the new value up the tree.
        self._siftUp( self._count - 1 )

    # Extract the maximum value from the heap.
    def extract( self ):
        assert self._count > 0, "Cannot extract from an empty heap."
        # Save the root value and copy the last heap value to the root.
        value = self._elements[0]
        self._count -= 1
        self._elements[0] = self._elements[ self._count ]
        # Sift the root value down the tree.
        self._siftDown( 0 )
        return value

    # Sift the value at the ndx element up the tree.
    def _siftUp( self, ndx ):
        if ndx > 0 :
            parent = (ndx - 1) // 2
            if self._elements[ndx] > self._elements[parent] :
                self._elements[ndx],self._elements[parent] = self._elements[parent],self._elements[ndx]
                self._siftUp( parent )

    # Sift the value at the ndx element down the tree.
    def _siftDown( self, ndx ):
        left = 2 * ndx + 1
        right = 2 * ndx + 2
        # Determine which node contains the larger value.
        largest = ndx
        if left < self._count and self._elements[left] >= self._elements[largest] :
            largest = left
        if right < self._count and self._elements[right] >= self._elements[largest]:
            largest = right
        # If the largest value is not in the current node (ndx), swap it with
        # the largest value and repeat the process.
        if largest != ndx :
            self._elements[ndx],self._elements[largest] = self._elements[largest],self._elements[ndx]
            self._siftDown( largest )
    # Display the value in the heap
    def __str__(self):
        import math
        depth = int(math.log(self._count,2))
        print("depth: ",depth, " element: ",self._count)
        s = ""
        for i in range(depth + 1):
            for j in range(2**i):
                if (2 ** i ) - 1 + j < self._count:
                    s = s +str(self._elements[(2 ** i) - 1 + j])+ "\t"
            s = s + "\n"
        return s

    def __repr__(self):
        import math
        depth = int(math.log(self._count,2))
        print("depth: ",depth, " element: ",self._count)
        s = ""
        for i in range(depth + 1):
            for j in range(2**i):
                if (2 ** i ) - 1 + j < self._count:
                    s = s +str(self._elements[(2 ** i) - 1 + j])+ "\t"
            s = s + "\n"
        return s
    
class MinHeap:
    # Create a min-heap with maximum capacity of maxSize.
    def __init__(self, maxSize):
        self._elements = Array(maxSize)
        self._count = 0

    # Return the number of items in the heap.
    def __len__(self):
        return self._count

    # Return the maximum capacity of the heap.
    def capacity(self):
        return len(self._elements)

    # Add a new value to the heap.
    def add(self, value):
        assert self._count < self.capacity(), "Cannot add to a full heap."
        self._elements[self._count] = value
        self._count += 1
        self._siftUp(self._count - 1)

    # Extract the minimum value from the heap.
    def extract(self):
        assert self._count > 0, "Cannot extract from an empty heap."
        value = self._elements[0]
        self._count -= 1
        self._elements[0] = self._elements[self._count]
        # (คงพฤติกรรมเดิม: ไม่แตะค่า slot ท้าย)
        self._siftDown(0)
        return value

    # Sift the value at the ndx element up the tree 
    def _siftUp(self, ndx):
        if ndx > 0:
            parent = (ndx - 1) // 2
            if self._elements[ndx] < self._elements[parent]:   
                self._elements[ndx], self._elements[parent] = self._elements[parent], self._elements[ndx]
                self._siftUp(parent)

    # Sift the value at the ndx element down the tree 
    def _siftDown(self, ndx):
        left = 2 * ndx + 1
        right = 2 * ndx + 2
        smallest = ndx
        if left < self._count and self._elements[left] <= self._elements[smallest]:  
            smallest = left
        if right < self._count and self._elements[right] <= self._elements[smallest]: 
            smallest = right
        if smallest != ndx:
            self._elements[ndx], self._elements[smallest] = self._elements[smallest], self._elements[ndx]
            self._siftDown(smallest)

    # Display the value in the heap 
    def __str__(self):
        import math
        depth = int(math.log(self._count, 2))
        print("depth: ", depth, " element: ", self._count)
        s = ""
        for i in range(depth + 1):
            for j in range(2**i):
                if (2 ** i) - 1 + j < self._count:
                    s = s + str(self._elements[(2 ** i) - 1 + j]) + "\t"
            s = s + "\n"
        return s

    def __repr__(self):
        import math
        depth = int(math.log(self._count, 2))
        print("depth: ", depth, " element: ", self._count)
        s = ""
        for i in range(depth + 1):
            for j in range(2**i):
                if (2 ** i) - 1 + j < self._count:
                    s = s + str(self._elements[(2 ** i) - 1 + j]) + "\t"
            s = s + "\n"
        return s

def heap_sort_asc(seq):
    
    n = len(seq)
    h = MaxHeap(n)

    for i in range(n):
        h.add(seq[i])
   
    for i in range(n - 1, -1, -1):
        seq[i] = h.extract()
    return seq

def is_sorted_nondecreasing(seq):
    for i in range(1, len(seq)):
        if seq[i-1] > seq[i]:
            return False
    return True

class TreeNode:
    def __init__(self, key, value, leftChild=None, rightChild=None, parent=None):
        self.key = key
        self.payload = value
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.parent = parent
        self.balanceFactor = 0
#### Additional method ####
    def hasLeftChild(self):
        return self.leftChild is not None

    def hasRightChild(self):
        return self.rightChild is not None

    def isLeftChild(self):
        return (self.parent and self.parent.leftChild) == self

    def isRightChild(self):
        return (self.parent and self.parent.rightChild) == self

    def isRoot(self):
        return self.parent is None

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return (self.rightChild is not None) or (self.leftChild is not None)

    def hasBothChildren(self):
        return (self.rightChild is not None) and (self.leftChild is not None)

    def replaceNodeData(self, key, value, leftChild, rightChild):
        self.key = key
        self.payload = value
        self.leftChild = leftChild
        self.rightChild = rightChild
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

    def __repr__(self):
        return str((self.key,self.payload))
    
class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def __len__(self):
        return self.size
#----------------------------------#    
    def __setitem__(self, k, v):
        self.put(k, v)

    def put(self, key, val):
        if self.root: # There is already root then add it as child node
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val) # if no root then promote it as root
            self.size += 1

    def _put(self, key, val, currentNode):
        if key == currentNode.key:  # Check the position
            currentNode.replaceNodeData(key,val,currentNode.leftChild,currentNode.rightChild) # if the same key then update new value
        else:
            if key < currentNode.key: # if the (new)key is less than the currentNode.key then add it as left child
                if currentNode.hasLeftChild(): # if there is already left child then add it as another left child
                    self._put(key, val, currentNode.leftChild)
                else:
                    currentNode.leftChild = TreeNode(key, val, parent=currentNode) #if there is no left child then add a new left child
                    self.size += 1
            else:
                if currentNode.hasRightChild():
                    self._put(key, val, currentNode.rightChild)
                else:
                    currentNode.rightChild = TreeNode(key, val, parent=currentNode)
                    self.size += 1
#----------------------------------#    
    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self, key, currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)
#----------------------------------#    
    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False
#----------------------------------#    
    def delete(self, key):
        if self.size > 1:
            nodeToRemove = self._get(key, self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size-1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self, key):
        self.delete(key)

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                if self.isLeftChild():
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def findMin(self):
        current = self.root
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def remove(self, currentNode):
        if currentNode.isLeaf():  # leaf
            if currentNode == currentNode.parent.leftChild:
                currentNode.parent.leftChild = None
            else:
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren():  # interior
            succ = currentNode.findSuccessor()
            succ.spliceOut()
            currentNode.key = succ.key
            currentNode.payload = succ.payload

        else:  # this node has one child
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.leftChild
                elif currentNode.isRightChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.leftChild
                else:
                    currentNode.replaceNodeData(currentNode.leftChild.key,
                                                currentNode.leftChild.payload,
                                                currentNode.leftChild.leftChild,
                                                currentNode.leftChild.rightChild)
            else:
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key,
                                                currentNode.rightChild.payload,
                                                currentNode.rightChild.leftChild,
                                                currentNode.rightChild.rightChild)
    def inorder(self):
        result = []
        self._inorder_traversal(self.root, result)
        return result

    def _inorder_traversal(self, node, result):
        if node:
            self._inorder_traversal(node.leftChild, result)
            result.append((node.key, node.payload))
            self._inorder_traversal(node.rightChild, result)

class AVL(BinarySearchTree):
    def __init__(self):
        super().__init__()

    def _put(self, key, val, currentNode):
        if key == currentNode.key:
            currentNode.replaceNodeData(key,val,currentNode.leftChild,currentNode.rightChild)
        else:
            if key < currentNode.key:
                if currentNode.hasLeftChild():
                    self._put(key, val, currentNode.leftChild)
                else:
                    currentNode.leftChild = TreeNode(key, val, parent=currentNode)
                    self.size += 1
                    self.updateBalance(currentNode.leftChild)
            else:
                if currentNode.hasRightChild():
                    self._put(key, val, currentNode.rightChild)
                else:
                    currentNode.rightChild = TreeNode(key, val, parent=currentNode)
                    self.size += 1
                    self.updateBalance(currentNode.rightChild)

    def updateBalance(self, node):
        if node.balanceFactor > 1 or node.balanceFactor < -1:
            self.rebalance(node)
            return
        if node.parent is not None:
            if node.isLeftChild():
                node.parent.balanceFactor += 1
            elif node.isRightChild():
                node.parent.balanceFactor -= 1
            if node.parent.balanceFactor != 0:
                self.updateBalance(node.parent)

    def rebalance(self, node):
        if node.balanceFactor < 0:
            if node.rightChild.balanceFactor > 0:
                self.rotateRight(node.rightChild)
                self.rotateLeft(node)
            else:
                self.rotateLeft(node)
        elif node.balanceFactor > 0:
            if node.leftChild.balanceFactor < 0:
                self.rotateLeft(node.leftChild)
                self.rotateRight(node)
            else:
                self.rotateRight(node)
    # RolateLeft
    def rotateLeft(self, rotRoot):
        newRoot = rotRoot.rightChild
        rotRoot.rightChild = newRoot.leftChild
        if newRoot.leftChild is not None:
            newRoot.leftChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isLeftChild():
                rotRoot.parent.leftChild = newRoot
            else:
                rotRoot.parent.rightChild = newRoot
        newRoot.leftChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(rotRoot.balanceFactor, 0)
    # RolateRight
    def rotateRight(self, rotRoot):
        newRoot = rotRoot.leftChild
        rotRoot.leftChild = newRoot.rightChild
        if newRoot.rightChild is not None:
            newRoot.rightChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isRightChild():
                rotRoot.parent.rightChild = newRoot
            else:
                rotRoot.parent.leftChild = newRoot
        newRoot.rightChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(rotRoot.balanceFactor, 0)

class Graph:
    def __init__(self,maxVertices,directed=False):
        self._Vertices = list()
        self._MATRIX = Matrix(maxVertices,maxVertices)
        self._MATRIX.clear(None)
        self._directed = directed
#------------------------- Vertex class -----------------------
    class Vertex:
        __slots__ = '_element'

        def __init__(self, x):
            self._element = x

        def element(self):
            return self._element

        def __repr__(self):
            return str(self._element)
#------------------------- Edge class -------------------------
    class Edge:
        __slots__ = '_origin' , '_destination', '_element'

        def __init__(self, u, v, w):
            self._origin = u
            self._destination = v
            self._element = w

        def endpoints(self):
            return (self._origin, self._destination)

        def opposite(self, v):
            return self._destination if v is self._origin else self._origin

        def element(self):
            return self._element

        def __repr__(self):
            return str(self._element)
#-----------------------------------------------------------
    def is_directed(self):
        return self._directed

    def findindex(self,v):
        if v in self._Vertices:
            return self._Vertices.index(v)
#-------------------------------------------------------------------------
    def vertex_count(self):
        return len(self._Vertices)

    def vertices(self):
        return self._Vertices

    def edge_count(self):
        total = 0
        for row in range(self.vertex_count()):
            for col in range(self.vertex_count()):
                if self._MATRIX[row,col] != None:
                    total += 1
        return total if self.is_directed() else total // 2

    def edges(self):
        edges_list = list()
        for row in range(self.vertex_count()):
            for col in range(self.vertex_count()):
                if (self._MATRIX[row,col] not in edges_list) and (self._MATRIX[row,col] is not None):
                    edges_list.append(self._MATRIX[row,col])
        return edges_list

    def get_edge(self, u, v):
        return self._MATRIX[self.findindex(u),self.findindex(v)]

    def degree(self, v, outgoing=True):
        total = 0
        if outgoing:
            for col in range(self.vertex_count()):
                if self._MATRIX[self.findindex(v),col] != None:
                    total += 1
        #incoming
        else:
            for row in range(self.vertex_count()):
                if self._MATRIX[row,self.findindex(v)] != None:
                    total += 1

        return total

    def incident_edges(self, v, outgoing=True):
        adj = list()
        if outgoing:
            for col in range(self.vertex_count()):
                if self._MATRIX[self.findindex(v),col] != None:
                    adj.append(self._MATRIX[self.findindex(v),col])
        #incoming
        else:
            for row in range(self.vertex_count()):
                if self._MATRIX[row,self.findindex(v)] != None:
                    adj.append(self._MATRIX[row,self.findindex(v)])
        return adj
    def insert_vertex(self, x):
        v = self.Vertex(x)
        self._Vertices.append(v)
        return v

    def insert_edge(self, u, v, x):
        # u is origin
        # v is destination
        e = self.Edge(u, v, x)
        if self.is_directed():
            self._MATRIX[self.findindex(u),self.findindex(v)] = e
        else:
            self._MATRIX[self.findindex(u),self.findindex(v)] = e
            self._MATRIX[self.findindex(v),self.findindex(u)] = e
        return e

    def remove_vertex(self,v):
        i = self.findindex(v)
        if i is None:
            return None                     
        n = self.vertex_count()
        #------- เคลียร์แถว/คอลัมน์ของ v-----------
        for c in range(n): 
            self._MATRIX[i, c] = None
        for r in range(n):
            self._MATRIX[r, i] = None
        #------เลื่อนแถว ลงมาทับตำแหน่ง i (r=i+1..n-1 -> r-1)------------
        for r in range(i + 1, n):
            for c in range(n):
                self._MATRIX[r - 1, c] = self._MATRIX[r, c]
                self._MATRIX[r, c] = None

        #----เลื่อนคอลัมน์ มาทับตำแหน่ง i (c=i+1..n-1 -> c-1)
        for c in range(i + 1, n):
            for r in range(n):
                self._MATRIX[r, c - 1] = self._MATRIX[r, c]
                self._MATRIX[r, c] = None

        #------ลบ v ออกจากลิสต์เวอร์เท็กซ์ (index หลัง i จะลดลง 1 ตรงกับเมทริกซ์ที่เลื่อนแล้ว)
        self._Vertices.pop(i)
        return v
    
    def remove_edge(self, e):
        """ลบเส้นเชื่อม e ออกจากกราฟ (รองรับทั้ง directed/undirected)"""
        if e is None:
            return None
        u, v = e.endpoints()
        iu, iv = self.findindex(u), self.findindex(v)
        if iu is None or iv is None:
            return None                     # หรือ raise KeyError("vertex not in graph")
        if self._MATRIX[iu, iv] is None:
            return None                     # ไม่มี edge นี้อยู่แล้ว

        # ลบตำแหน่งหลัก
        self._MATRIX[iu, iv] = None
        # ถ้าเป็นกราฟไม่กำกับทิศ ต้องลบกลับอีกทิศด้วย
        if not self.is_directed():
            self._MATRIX[iv, iu] = None
        return e

    def DFS(self,u,discovered):
        print(u, end=' ')  
        for e in self.incident_edges(u):      
            v = e.opposite(u)                
            if v not in discovered:        
                discovered[v] = e             
                self.DFS(v, discovered) 


    def BFS(self, s, discovered):
        level = [s]                       
        print(s, end=' ')               
        while len(level) > 0:             
            next_level = []               
            for u in level:
                for e in self.incident_edges(u):  
                    v = e.opposite(u)
                    if v not in discovered:       
                        discovered[v] = e         
                        print(v, end=' ')          
                        next_level.append(v)      
            level = next_level      


    def transitive_closure(self):
        n = self.vertex_count()
        T = Matrix(n, n)
        T.clear(False)

        # เริ่มจากเมทริกซ์ขอบ (มีเส้นทางตรงถือว่า reachable)
        for i in range(n):
            for j in range(n):
                T[i, j] = (self._MATRIX[i, j] is not None)
            T[i, i] = True   # ถึงตัวเองเสมอ

        # Warshall: O(n^3)
        for k in range(n):
            for i in range(n):
                if T[i, k]:                  # pruning เล็กน้อย
                    for j in range(n):
                        if T[k, j]:
                            T[i, j] = True
        return T

    def __repr__(self):
        s = '['
        for r in range(self._MATRIX.numRows()):
            for c in self._MATRIX._theRows[r]:
                if c is None:
                    c = 0
                s = s + str(c) + ', '
            s = s[:-2] + ' \n '
        s = s[:-3] + ' ]'
        return s
    
def DFS(g,u,discovered):
    print(u)
    for e in g.incident_edges(u):
        v = e.opposite(u)
        if v not in discovered:
            discovered[v] = e
            DFS(g,v,discovered)

def BFS(g, s, discovered):
    level = [s]
    while len(level) > 0:
        next_level= []
        for u in level:
            print(u)
            for e in g.incident_edges(u):
                v = e.opposite(u)
                if v not in discovered:
                    discovered[v] = e
                    next_level.append(v)
            level = next_level

class PriorityQueueBase:
    # Abstract base class for a priority queue.
    class Item:
        __slots__ = '_key', '_value'

        def __init__(self, k, v):
            self._key = k
            self._value = v

        def __lt__(self, other):
            return self._key < other._key  # compare items based on their keys

        def is_empty(self):  # concrete method assuming abstract len
            return len(self) == 0
# -------------------------------------------


class HeapPriorityQueue(PriorityQueueBase):  # base class defines Item
    # Use Heap to implement PriorityQueue
    def _parent(self, j):
        return (j - 1) // 2

    def _left(self, j):
        return 2*j + 1

    def _right(self, j):
        return 2*j + 2

    def _has_left(self, j):
        return self._left(j) < len(self._data)  # index beyond end of list?

    def _has_right(self, j):
        return self._right(j) < len(self._data)  # index beyond end of list?

    def _swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def _siftup(self, j):
        parent = self._parent(j)
        if j > 0 and self._data[j] < self._data[parent]:
            self._swap(j, parent)
            self._siftup(parent)  # recur at position of parent

    def _siftdown(self, j):
        if self._has_left(j):
            left = self._left(j)
            small_child = left  # although right may be smaller
            if self._has_right(j):
                right = self._right(j)
                if self._data[right] < self._data[left]:
                    small_child = right
            if self._data[small_child] < self._data[j]:
                self._swap(j, small_child)
                self._siftdown(small_child)  # recur at position of small child

    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):  # concrete method assuming abstract len
        return len(self) == 0

    def add(self, key, value):
        self._data.append(self.Item(key, value))
        self._siftup(len(self._data) - 1)  # upheap newly added position

    def min(self):
        if self.is_empty():
            raise Empty('Priority queue is empty.')
        item = self._data[0]
        return (item._key, item._value)

    def remove_min(self):
        if self.is_empty():
            raise Empty('Priority queue is empty.')
        self._swap(0, len(self._data) - 1)  # put minimum item at the end
        item = self._data.pop()  # and remove it from the list;
        self._siftdown(0)  # then fix new root
        return (item._key, item._value)
# ---------------------------------------------------------------------------------


class AdaptableHeapPriorityQueue(HeapPriorityQueue):
    # A locator-based priority queue implemented with a binary heap.
    # ------------------------------ nested Locator class --------------------------
    class Locator(HeapPriorityQueue.Item):
        # Token for locating an entry of the priority queue.
        __slots__ = '_index'  # add index as additional field

        def __init__(self, k, v, j):
            super().__init__(k, v)
            self._index = j
# ------------------------------ nonpublic behaviors ------------------------------
# override swap to record new indices

    def _swap(self, i, j):
        super()._swap(i, j)  # perform the swap
        self._data[i]._index = i  # reset locator index (post-swap)
        self._data[j]._index = j  # reset locator index (post-swap)

    def _bubble(self, j):
        if j > 0 and self._data[j] < self._data[self._parent(j)]:
            self._siftup(j)
        else:
            self._siftdown(j)

    def add(self, key, value):
        # Add a key-value pair
        token = self.Locator(key, value, len(self._data))  # initiaize locator index
        self._data.append(token)
        self._siftup(len(self._data) - 1)
        return token

    def update(self, loc, newkey, newval):
        # Update the key and value for the entry identified by Locator loc
        j = loc._index
        if not (0 <= j < len(self) and self._data[j] is loc):
            raise ValueError('Invalid locator')
        loc._key = newkey
        loc._value = newval
        self._bubble(j)

    def remove(self, loc):
        # Remove and return the (k,v) pair identified by Locator loc.”””
        j = loc._index
        if not (0 <= j < len(self) and self._data[j] is loc):
            raise ValueError('Invalid locator')
        if j == len(self) - 1:  # item at last position
            self._data.pop()  # just remove it
        else:
            self._swap(j, len(self)-1)  # swap item to the last position
            self._data.pop()  # remove it from the list
            self._bubble(j)  # fix item displaced by the swap
        return (loc._key, loc._value)
    
def Dijkstra(G, s):
    d = { } # d[v] is upper bound from s to v
    cloud = { } # map reachable v to its d[v] value
    PQ = AdaptableHeapPriorityQueue( ) # vertex v will have key d[v]
    pqlocator = { } # map from vertex to its pq locator

    # for each vertex v of the graph, add an entry to the priority queue, with
    # the source having distance 0 and all others having infinite distance
    for v in G.vertices( ):
        if v is s:
            d[v] = 0
        else:
            d[v] = float('inf') # syntax for positive infinity
        pqlocator[v] = PQ.add(d[v], v) # save locator for future updates

    while not PQ.is_empty( ):
        key, u = PQ.remove_min() # key is Priority(weight), u is vertex
        cloud[u] = key
        #cloud[u] = key # its correct d[u] value
        del pqlocator[u] # u is no longer in pq
        for e in G.incident_edges(u): # outgoing edges (u,v)
            v = e.opposite(u)
            if v not in cloud:
                # perform relaxation step on edge (u,v)
                wgt = e.element( )
                if d[u] + wgt < d[v]: # better path to v?
                    d[v] = d[u] + wgt # update the distance
                    PQ.update(pqlocator[v], d[v], v) # update the pq entry
    return cloud # only includes reachable vertices

def Floyd_Warshall(g):
    n = g.vertex_count()
    vertices = g.vertices()
    
    D = Matrix(n, n)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = 0 
            else:
                edge = g.get_edge(vertices[i], vertices[j])
                
                if edge is not None:
                 D[i, j] = edge.element()
                else:
                    D[i, j] = float('inf')

    for k in range(n):
        for i in range(n):
            for j in range(n):
                
                path_via_k = D[i, k] + D[k, j]
                
                if D[i, j] > path_via_k:
                    D[i, j] = path_via_k
    
    return D

def PrimJarnik(g):
#Compute a minimum spanning tree of weighted graph g.
#Return a list of edges that comprise the MST (in arbitrary order).
    d = { } # d[v] is bound on distance to tree
    tree = {  } # edges in spanning tree
    pq = AdaptableHeapPriorityQueue( ) # d[v] maps to value (v, e=(u,v))
    pqlocator = { } # map from vertex to its pq locator
    # for each vertex v of the graph, add an entry to the priority queue, with
    # the source having distance 0 and all others having infinite distance
    for v in g.vertices( ):
        if len(d) == 0: # this is the first node
            d[v] = 0 # make it the root
        else:
            d[v] = float('inf') # positive infinity
        pqlocator[v] = pq.add(d[v], (v,None))
    while not pq.is_empty( ):
        key,value = pq.remove_min()
        u,edge = value # unpack tuple from pq
        tree[u] = edge
        del pqlocator[u] # u is no longer in pq
        for link in g.incident_edges(u):
            v = link.opposite(u)
            if v in pqlocator: # thus v not yet in tree
                # see if edge (u,v) better connects v to the growing tree
                wgt = link.element( )
                if wgt < d[v]: # better edge to v?
                    d[v] = wgt # update the distance
                    pq.update(pqlocator[v], d[v], (v, link)) # update the pq entry
    return tree

class UnionFind:

    class Position:
        __slots__ = '_container' , '_element' , '_size' , '_parent'
        def __init__(self, container, e):
            self._container = container # reference to UnionFind instance
            self._element = e
            self._size = 1
            self._parent = self # convention for a group leader
        def element(self):
            return self._element
#------------------------- Union-find -------------------------
    def make_group(self, e):
        return self.Position(self, e)
    
    def find(self, p):
        
        if p._parent != p:
            p._parent = self.find(p._parent)
        return p._parent
        
    def union(self, p, q):
        
        a = self.find(p)
        b = self.find(q)
        if a is not b:
            if a._size >= b._size:
                b._parent = a
                a._size += b._size
            else:
                a._parent = b
                b._size += a._size

def Kruskal(g):
    tree = { } # Dictionary of edges in spanning tree
    pq = HeapPriorityQueue( ) # entries are edges in G, with weights as key
    forest = UnionFind( ) # keeps track of forest clusters
    position = { } # map each node to its Partition entry
    for v in g.vertices( ):
        position[v] = forest.make_group(v)
    for e in g.edges( ):
        pq.add(e.element( ), e) # edge’s element is assumed to be its weight
    size = g.vertex_count( )
    while len(tree) != size - 1 and not pq.is_empty():
        # tree not spanning and unprocessed edges remain
        weight,edge = pq.remove_min()
        u,v = edge.endpoints( )
        a = forest.find(position[u])
        b = forest.find(position[v])
        if a != b:
            tree[edge.endpoints()] = edge
            forest.union(a,b)
    return tree

def calculate_mst_weight(mst_tree):#input graph ที่แปลงเป็น prim jarnik หรือ kruskal แล้ว
    total_weight = 0
    for edge in mst_tree.values():
        if edge is not None:
            total_weight += edge.element()
    return total_weight

def Dijkstra_wieght(shortest_paths):#input graph ที่แปลงเป็น dijkstra แล้ว
    w=0
    for i in shortest_paths:
        x=shortest_paths[i]
        w=w+x
    return w