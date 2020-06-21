import numpy as np
import math


def deg_to_rad(deg):
  return deg * np.pi / 180.0


def _data(v, dtype):
  #if isinstance(v, np.ndarray);
  #  return v
  if isinstance(v, MData):
    return v._data
  if isinstance(v, list):
    return np.array(v, dtype=dtype)
  return v

def _data_assign(lhs, rhs):
  if rhs.ndim == 1:
    lhs[0:len(rhs)] = rhs[:]
  else:
    for i in range(len(rhs)):
      _data_assign(lhs[i], rhs[i])

class MData:
  def __init__(self, data):
    self._data = data

  def __matmul__(self, v): return self.__class__(self._data @ _data(v, self._data.dtype))

  def __add__(self, v): return self.__class__(self._data + _data(v, self._data.dtype))
  def __sub__(self, v): return self.__class__(self._data - _data(v, self._data.dtype))
  def __mul__(self, v): return self.__class__(self._data * _data(v, self._data.dtype))
  def __radd__(self, v): return self.__class__(self._data + _data(v, self._data.dtype))
  def __rsub__(self, v): return self.__class__(self._data - _data(v, self._data.dtype))
  def __rmul__(self, v): return self.__class__(self._data * _data(v, self._data.dtype))
  def __neg__(self): return self.__class__(- self._data)
  def __pos__(self): return self.__class__(+ self._data)
  def __abs__(self): return self.__class__(abs(self._data))
  #def __trunc__(self): return self.__class__(math.trunc(self._data))

  def __getitem__(self, i):
    return self._data[i]

  def __setitem__(self, i, value):
    self._data[i] = value

  def assign(self, rhs):
    self[:] = _data(rhs, self._data.dtype)[:]
    #_data_assign(self._data, _data(rhs, self._data.dtype))

  @property
  def data(self):
    return self._data[:]


class MVec3(MData):
  def __init__(self, *args):
    super().__init__(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    if len(args) == 1:
      self._data[:] = _data(args[0], self._data.dtype)
    elif len(args) == 3:
      self._data[0] = args[0]
      self._data[1] = args[1]
      self._data[2] = args[2]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  @staticmethod
  def from_xyz(x, y, z):
    return MVec3(x, y, z)

  @staticmethod
  def from_vec3(v):
    return MVec3(v.x, v.y, v.z)

  def __str__(self):
    return str(self._data)

  def __repr__(self):
    return '<MVec3 {}>'.format(repr(self._data))

  @property
  def x(self):
    return self._data[0]

  @x.setter
  def x(self, value):
    self._data[0] = value

  @property
  def y(self):
    return self._data[1]

  @y.setter
  def y(self, value):
    self._data[1] = value

  @property
  def z(self):
    return self._data[2]

  @z.setter
  def z(self, value):
    self._data[2] = value

  def dot(self, v):
    return np.dot(self._data, _data(v, self._data.dtype))

  @property
  def mag_sq(self):
    x = self._data[0]
    y = self._data[1]
    z = self._data[2]
    return x*x + y*y + z*z

  @property
  def mag(self):
    return np.sqrt(self.mag_sq)

  def normalize(self):
    invMag = 1.0 / self.mag
    self.x *= invMag
    self.y *= invMag
    self.z *= invMag

  def normalized(self):
    r = self.__class__(self)
    r.normalize()
    return r

  def cross(self, v):
    _data = self._data
    return MVec3( _data[ 1 ]*v._data[ 2 ] - _data[ 2 ]*v._data[ 1 ],
                  _data[ 2 ]*v._data[ 0 ] - _data[ 0 ]*v._data[ 2 ],
                  _data[ 0 ]*v._data[ 1 ] - _data[ 1 ]*v._data[ 0 ] );


class MMat3x3(MData):
  def __init__(self, *args):
    super().__init__(np.eye(3))
    if len(args) == 1:
      v = args[0]
      self.assign(v)
    elif len(args) == 9:
      self._data[ 0, 0 ] = args[ 0 ]
      self._data[ 0, 1 ] = args[ 1 ]
      self._data[ 0, 2 ] = args[ 2 ]
      self._data[ 1, 0 ] = args[ 3 ]
      self._data[ 1, 1 ] = args[ 4 ]
      self._data[ 1, 2 ] = args[ 5 ]
      self._data[ 2, 0 ] = args[ 6 ]
      self._data[ 2, 1 ] = args[ 7 ]
      self._data[ 2, 2 ] = args[ 8 ]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MMat3x3\n{}>".format(self._data)

  def __getitem__(self, idx):
    #i, j = idx
    #assert i >= 0 and i < 4
    #assert j >= 0 and j < 4
    #return self._data[ i*4 + j ]
    r = self._data[ idx ]
    if isinstance(idx, int):
      r = MVec3(r)
    return r

  def __setitem__(self, idx, value):
    #i, j = idx
    #assert i >= 0 and i < 4
    #assert j >= 0 and j < 4
    #self._data[ i*4 + j ] = value
    self._data[ idx ] = _data(value, self._data.dtype)

  def get_at(self, i):
    return self._data[i // 3][i % 3]

  def set_at(self, i, value):
    self._data[i // 3][i % 3] = value

  def get_col(self, i):
    return MVec3(
        self._data[ 0 ][ i ],
        self._data[ 1 ][ i ],
        self._data[ 2 ][ i ])

  def set_col(self, i, col):
    col = _data( col, self._data.dtype )
    self._data[ 0 ][ i ] = col[0]
    self._data[ 1 ][ i ] = col[1]
    self._data[ 2 ][ i ] = col[2]

  def transpose(self):
    self._data[0][1], self._data[1][0] = self._data[1][0], self._data[0][1]
    self._data[0][2], self._data[2][0] = self._data[2][0], self._data[0][2]
    self._data[0][3], self._data[3][0] = self._data[3][0], self._data[0][3]

  def transposed(self):
    return self.__class__(
        self._data[0][0], self._data[1][0], self._data[2][0],
        self._data[0][1], self._data[1][1], self._data[2][1],
        self._data[0][2], self._data[1][2], self._data[2][2])

  def __mul__(self, m):
    return self.__class__(np.matmul(self._data, _data(m, self._data.dtype)))

  def rotate_point(self, point):
    return MVec3(
      self.get_at( 0 ) * point.x + self.get_at( 1 ) * point.y + self.get_at( 2 ) * point.z,
      self.get_at( 3 ) * point.x + self.get_at( 4 ) * point.y + self.get_at( 5 ) * point.z,
      self.get_at( 6 ) * point.x + self.get_at( 7 ) * point.y + self.get_at( 8 ) * point.z )

  def rotate_point_fast(self, point):
    x = self.get_at( 0 ) * point.x + self.get_at( 1 ) * point.y + self.get_at( 2 ) * point.z;
    y = self.get_at( 3 ) * point.x + self.get_at( 4 ) * point.y + self.get_at( 5 ) * point.z;
    z = self.get_at( 6 ) * point.x + self.get_at( 7 ) * point.y + self.get_at( 8 ) * point.z;
    point.x = x
    point.y = y
    point.z = z


class MMat4x4(MData):
  def __init__(self, *args):
    super().__init__(np.eye(4))
    if len(args) == 1:
      v = _data(args[0], self._data.dtype)
      if v.ndim == 2 and np.prod(v.shape) == 9:
        rot = v
        self._data[ 0, 0 ] = rot[ 0, 0 ]
        self._data[ 0, 1 ] = rot[ 0, 1 ]
        self._data[ 0, 2 ] = rot[ 0, 2 ]

        self._data[ 1, 0 ] = rot[ 1, 0 ]
        self._data[ 1, 1 ] = rot[ 1, 1 ]
        self._data[ 1, 2 ] = rot[ 1, 2 ]

        self._data[ 2, 0 ] = rot[ 2, 0 ]
        self._data[ 2, 1 ] = rot[ 2, 1 ]
        self._data[ 2, 2 ] = rot[ 2, 2 ]
      else:
        self.assign(v)
    elif len(args) == 2:
      rot = _data(args[0], self._data.dtype)
      pos = _data(args[1], self._data.dtype)
      self._data[ 0, 0 ] = rot[ 0, 0 ]
      self._data[ 0, 1 ] = rot[ 0, 1 ]
      self._data[ 0, 2 ] = rot[ 0, 2 ]
      self._data[ 0, 3 ] = pos[ 0 ]

      self._data[ 1, 0 ] = rot[ 1, 0 ]
      self._data[ 1, 1 ] = rot[ 1, 1 ]
      self._data[ 1, 2 ] = rot[ 1, 2 ]
      self._data[ 1, 3 ] = pos[ 1 ]

      self._data[ 2, 0 ] = rot[ 2, 0 ]
      self._data[ 2, 1 ] = rot[ 2, 1 ]
      self._data[ 2, 2 ] = rot[ 2, 2 ]
      self._data[ 2, 3 ] = pos[ 2 ]

    elif len(args) == 16:
      self._data[ 0, 0 ] = args[ 0 ]
      self._data[ 0, 1 ] = args[ 1 ]
      self._data[ 0, 2 ] = args[ 2 ]
      self._data[ 0, 3 ] = args[ 3 ]

      self._data[ 1, 0 ] = args[ 4 ]
      self._data[ 1, 1 ] = args[ 5 ]
      self._data[ 1, 2 ] = args[ 6 ]
      self._data[ 1, 3 ] = args[ 7 ]

      self._data[ 2, 0 ] = args[ 8 ]
      self._data[ 2, 1 ] = args[ 9 ]
      self._data[ 2, 2 ] = args[ 10 ]
      self._data[ 2, 3 ] = args[ 11 ]

      self._data[ 3, 0 ] = args[ 12 ]
      self._data[ 3, 1 ] = args[ 13 ]
      self._data[ 3, 2 ] = args[ 14 ]
      self._data[ 3, 3 ] = args[ 15 ]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MMat4x4\n{}>".format(self._data)

  def __getitem__(self, idx):
    #i, j = idx
    #assert i >= 0 and i < 4
    #assert j >= 0 and j < 4
    #return self._data[ i*4 + j ]
    return self._data[ idx ]

  def __setitem__(self, idx, value):
    #i, j = idx
    #assert i >= 0 and i < 4
    #assert j >= 0 and j < 4
    #self._data[ i*4 + j ] = value
    self._data[ idx ] = _data(value, self._data.dtype)

  def get_at(self, i):
    return self._data[i // 4][i % 4]

  def set_at(self, i, value):
    self._data[i // 4][i % 4] = value

  def __mul__(self, m):
    return self.__class__(np.matmul(self._data, _data(m, self._data.dtype)))

  def inverse(self, output):
    try:
      m = np.linalg.inv(self._data)
      output.assign(m)
      return True
    except np.linalg.LinAlgError:
      return False


class MPlane:
  def __init__(self, *args):
    self._normal = MVec3(0.0, 1.0, 0.0)
    self._d = 0.0
    if len(args) == 1:
      v = args[0]
      if isinstance(v, MPlane):
        self._normal[:] = v._normal[:]
        self._d = v._d
      else:
        raise ValueError("Unknown argument type {}".format(v))
    elif len(args) == 2:
      #if not isinstance(args[0], MVec3):
      #  raise ValueError("Argument 0: expected MVec3, got {}".format(args[0]))
      self._normal.assign(args[0])
      self._normal.normalize()
      if isinstance(args[1], float) or isinstance(args[1], int):
        self._d = args[1]
      else:
        point = _data(args[1], self._normal._data.dtype)
        self._d = -self._normal.dot(point)
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def assign(self, rhs):
    self._normal.assign(rhs._normal)
    self._d = rhs._d

  def __repr__(self):
    return "<MPlane normal=({}, {}, {}) d={}>".format(self._normal.x, self._normal.y, self._normal.z, self._d)

  @property
  def normal(self):
    return self._normal

  @normal.setter
  def normal(self, value):
    self._normal.assign(value)

  @property
  def d(self):
    return self._d

  @d.setter
  def d(self, value):
    assert isinstance(value, float) or isinstance(value, int)
    self._d = float(value)

  def dist(self, point):
    return self._normal.dot(point) + self._d


class GrProjection:
  def __init__(self):
    self._fov = 0.0
    self._near_clip = MPlane( MVec3( 0.0, 0.0, -1.0 ), MVec3( 0.0, 0.0, -1.0 ) )
    self._far_dist = 1000.0
    self._aspect = 1.0
    self._left = -1.0
    self._right = 1.0
    self._bottom = -1.0
    self._top = 1.0
    self._mat = MMat4x4()
    self._dirty = True

  def assign(self, rhs):
    self._fov = rhs._fov
    self._near_clip.assign(rhs._near_clip)
    self._far_dist = rhs._far_dist
    self._aspect = rhs._aspect
    self._left = rhs._left
    self._right = rhs._right
    self._bottom = rhs._bottom
    self._top = rhs._top
    self._mat.assign(rhs._mat)
    self._dirty = rhs._dirty

  @classmethod
  def perspective(cls, fov, far_dist, aspect, near_clip_plane):
    self = cls()
    self._fov = fov
    self._far_dist = far_dist
    self._aspect = aspect
    self._near_clip.assign(near_clip_plane)
    assert self._fov >= 0.0
    return self

  def __repr__(self):
    return "<GrProjection fov={} near_clip={} far_dist={} aspect={} left={} right={} bottom={} top={}>".format(self._fov, self._near_clip, self._far_dist, self._aspect, self._left, self._right, self._bottom, self._top)

  @property
  def is_ortho(self):
    return self._fov == 0.0

  @property
  def fov(self):
    return self._fov

  @property
  def width(self):
    return self._right - self._left

  @property
  def height(self):
    return self._top - self._bottom

  @property
  def near_clip(self):
    return self._near_clip

  @property
  def far_dist(self):
    return self._far_dist

  @property
  def aspect(self):
    return self._aspect

  @property
  def left(self):
    return self._left

  @property
  def right(self):
    return self._right

  @property
  def top(self):
    return self._top

  @property
  def bottom(self):
    return self._bottom

  @property
  def matrix(self):
    if self._dirty:
      assert abs(self._right - self._left) > 0.0001
      assert abs(self._top - self._bottom) > 0.0001

      xScale = 2.0 / ( self._right - self._left )
      yScale = 2.0 / ( self._top - self._bottom )
      xOffset = ( self._right + self._left ) / ( self._right - self._left )
      yOffset = ( self._top + self._bottom ) / ( self._top - self._bottom )

      if not self.is_ortho:
        projNear = 1.0 / np.tan( self._fov * 0.5 )

        self._mat[ 0, 0 ] = xScale * projNear / ( self._aspect )
        self._mat[ 0, 1 ] = 0.0
        self._mat[ 0, 2 ] = xOffset
        self._mat[ 0, 3 ] = 0.0

        self._mat[ 1, 0 ] = 0.0
        self._mat[ 1, 1 ] = yScale * projNear
        self._mat[ 1, 2 ] = yOffset
        self._mat[ 1, 3 ] = 0.0

        self._mat[ 2, 0 ] = 0.0
        self._mat[ 2, 1 ] = 0.0
        self._mat[ 2, 2 ] = -1.0 # ( projNear + _farDist ) / ( projNear - _farDist )
        self._mat[ 2, 3 ] = -1.0 # _farDist * projNear / ( projNear - _farDist )

        self._mat[ 3, 0 ] = 0.0
        self._mat[ 3, 1 ] = 0.0
        self._mat[ 3, 2 ] = -1.0
        self._mat[ 3, 3 ] = 0.0

        # incorperate the near clip plane into the matrix by applying a shear
        # on the Z axis.
        self.add_near_clip();
      else:
        assert self._far_dist > 0.01

        self._mat[ 0, 0 ] = xScale
        self._mat[ 0, 1 ] = 0.0
        self._mat[ 0, 2 ] = 0.0
        self._mat[ 0, 3 ] = -xOffset

        self._mat[ 1, 0 ] = 0.0
        self._mat[ 1, 1 ] = yScale
        self._mat[ 1, 2 ] = 0.0
        self._mat[ 1, 3 ] = -yOffset

        self._mat[ 2, 0 ] = 0.0
        self._mat[ 2, 1 ] = 0.0
        self._mat[ 2, 2 ] = ( -1.0 / self._far_dist)
        self._mat[ 2, 3 ] = 0.0

        self._mat[ 3, 0 ] = 0.0
        self._mat[ 3, 1 ] = 0.0
        self._mat[ 3, 2 ] = 0.0
        self._mat[ 3, 3 ] = 1.0

      # clear the dirty flag.
      self._dirty = False

    # return the cached matrix.
    return self._mat

  def add_near_clip(self):
    clipNormal = self._near_clip.normal
    self._mat[ 2, 0 ] = clipNormal.x
    self._mat[ 2, 1 ] = clipNormal.y
    self._mat[ 2, 2 ] = clipNormal.z
    self._mat[ 2, 3 ] = self._near_clip.d


class GrCamera:
  def __init__(self):
    self._proj = GrProjection.perspective( deg_to_rad(90.0), 1000.0, 1.0, MPlane( MVec3( 0.0, 0.0, -1.0 ), MVec3( 0.0, 0.0, -1.0 ) ) )
    self._pos = MVec3()
    self._rot = MMat3x3()
    self._far_cull = 1000.0
    self._view_proj_matrix = MMat4x4()
    self._view_matrix = MMat4x4()
    self._inv_view_matrix = MMat4x4()
    self._normal_rot = MMat3x3()
    #self._frustum = GrFrustum()
    self._dirty = True

  def assign(self, rhs):
    self.pos = rhs.pos
    self.rot = rhs.rot
    self.proj = rhs.proj

  @property
  def pos(self):
    return self._pos

  @pos.setter
  def pos(self, pos):
    self._pos.assign(pos)
    self._dirty = True

  @property
  def rot(self):
    return self._rot

  @rot.setter
  def rot(self, rot):
    self._rot.assign(rot)
    self._dirty = True

  @property
  def proj(self):
    return self._proj

  @proj.setter
  def proj(self, proj):
    self._proj.assign(proj)
    self._dirty = True

  @property
  def side_dir(self):
    return self._rot.get_col(0)

  @property
  def up_dir(self):
    return self._rot.get_col(1)

  @property
  def look_dir(self):
    return self._rot.get_col(2)

  def look_at(self, pos, target, world_up=None):
    zdir = pos - target
    assert zdir.mag_sq > 0.00001

    # mark as dirty.
    self._dirty = True

    # store the position.
    self._pos.assign(pos)

    # compute the new z basis.
    self.look(zdir, world_up=world_up)

  def look(self, tainted_dir, world_up=None):
    if world_up is None:
      world_up = [0.0, 1.0, 0.0]
    world_up = MVec3( world_up )
    world_up.normalize() # TODO: is this necessary?

    # mark as dirty.
    self._dirty = True

    # build the rotation matrix.
    look = tainted_dir.normalized()

    # compute the new x basis.
    if abs( look.y ) > 0.999:
      # compensate for trying to look directly up or down.
      right = MVec3(1, 0, 0)
    else:
      right = look.cross( world_up )
      right.normalize()

    # compute the new y basis.
    up = right.cross( look )

    # set the basis vectors.
    self._rot.set_col( 0, right )
    self._rot.set_col( 1, up )
    self._rot.set_col( 2, -look )

  def update_matrices(self):
    assert self._dirty

    # mark the matrices as up to date.
    self._dirty = False

    # create the projection matrix.
    projMatrix = self._proj.matrix

    transViewRot = self._rot.transposed()

    # create the view matrix.
    invCamPos = -transViewRot.rotate_point( self._pos )
    self._view_matrix = MMat4x4( transViewRot, invCamPos )

    # adjoint transpose.
    self._normal_rot = MMat3x3( transViewRot )

    # create the view-proj matrix.
    self._view_proj_matrix = projMatrix * self._view_matrix
    valid = self._view_matrix.inverse( self._inv_view_matrix )
    assert valid

    # reflection = False
    # # check to see if the view matrix is a reflection.
    # #if self._view_matrix[ 0, 0 ] * self._view_matrix[ 1, 1 ] * self._view_matrix[ 2, 2 ] < 0.0:
    # #  reflection = True;

    # # check to see if the view matrix is a reflection.
    # xAxis = MVec3( self._view_matrix[0][0:3] )
    # yAxis = MVec3( self._view_matrix[1][0:3] )
    # zAxis = MVec3( self._view_matrix[2][0:3] )
    
    # # do we have a reflection?
    # if xAxis.cross( yAxis ).dot( zAxis ) < 0.0:
    #   reflection = True;

  def build_world_matrix(self):
    return MMat4x4( self.rot, self.pos )

  @property
  def view_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._view_matrix

  @property
  def inv_view_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._inv_view_matrix

  @property
  def view_proj_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._view_proj_matrix

  @property
  def normal_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._normal_rot
    

