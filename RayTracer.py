#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
from PIL import Image, ImageFilter


# In[2]:


class CoordinateSystem:

    def __init__(self, anchor):
        up = np.array([anchor.x + 0.000001, anchor.y + 1, anchor.z])
        self.i = Vector(*np.cross(np.array(anchor.to_list()), up)).normalize(4)
        self.j = Vector(*np.cross(np.array(self.i.to_list()), np.array(anchor.to_list()))).normalize(4)
        self.k = Vector(*np.cross(np.array(self.j.to_list()), np.array(self.i.to_list()))).normalize(4)
        
    def __str__(self):
        return f'{str(self.i)}\n{str(self.j)}\n{str(self.k)}'


# In[3]:


def HasDecimalPlaces(number):
    return (number / 2) % 1 > 0


# In[4]:


def GeneratePixelGrid(rays_amt):
    side = math.sqrt(rays_amt)
    slices_side = 1 / side
    
    grid = np.zeros((0, 3))
    ind = 0

    for it in range(rays_amt):
        cell = it % side
        if(cell == 0): ind+=1

        x_start = cell * slices_side
        y_start = (ind-1) * slices_side

        x_end = (cell + 1) * slices_side
        y_end = ind * slices_side

        grid = np.vstack([grid, [random.uniform(x_start, x_end), random.uniform(y_start, y_end), 0]])

    return grid

def ToViewportPoint(center, point, width, height, distance):
    half_h = height / 2
    half_w = width / 2
    
    viewport_size = (360,180)
    
    x = center.x + (point.x - half_w) * (viewport_size[0]/width)
    y = center.y + (point.y - half_h) * (viewport_size[1]/height)
    z = center.z + distance
    
    return Point(x, y, z).to_list()
        
class Scene:
    """
    The scene that gets rendered. Contains information like the camera
    position, the different objects present, etc.
    """

    def __init__(self, camera, objects, lights, width, height, viewport_dist, rot_cam_x, rot_cam_y):
        self.camera = camera
        self.objects = objects
        self.lights = lights
        self.width = width
        self.height = height
        self.viewport_dist = viewport_dist
        self.rot_cam_x = rot_cam_x
        self.rot_cam_y = rot_cam_y

    def render(self, print_exception=False):        
        pixels = np.zeros((self.height, self.width, 3))
        
        for y in range(self.height):
#             print('row ', y, ' of ', self.height)
            for x in range(self.width):
                progress_max = self.height * self.width
                progress = ((y * self.width + x) / progress_max) * 100
                print('\r[{0}] {1:.2f}%'.format('#'*(int(progress//10)) + '_'*(int(10 - (progress//10))), progress), end='')
                
                grid = GeneratePixelGrid(16)
                
                noisy_points = grid + Point(x, y).to_list()
                noisy_ray_directions = [ToViewportPoint(
                    self.camera, 
                    Point(*noisy_pt), 
                    self.width, 
                    self.height, 
                    self.viewport_dist
                ) for noisy_pt in noisy_points.tolist()]
        
                colors = np.zeros((0, 3))
                
                for ray_direction in noisy_ray_directions:
#                     direction = Vector(*ray_direction) - self.camera
#                     origin = self.camera
#                     ray = Ray(origin, direction)
                    ray = Ray(Vector(*ray_direction))
                    ray.rotate_direction(self.rot_cam_x, self.rot_cam_y)
                    color = self._shoot_ray(ray)
                    colors = np.vstack([colors, color.to_list()])
                
                pixels[y, x, :] = np.mean(colors, axis=0)

        return pixels

    def _shoot_ray(self, ray, depth=0, max_depth=5, print_exception=True):
        try:
            color = Color()
        
            if depth >= max_depth: return color

            intersection = self._get_intersection(ray)
            if intersection is None:
                return color

            obj, dist = intersection
            intersection_pt = ray.point_at_dist(dist)
            surface_norm = obj.surface_norm(intersection_pt)

            obj_base_color = obj.get_color(intersection_pt)
            # ambient light
            color += obj_base_color * obj.material.ambient

            # lambert shading
            for light in self.lights:
                pt_to_light_vec = (light - intersection_pt).normalize()
                pt_to_light_ray = Ray(intersection_pt, pt_to_light_vec)
                if self._get_intersection(pt_to_light_ray) is None:
                    lambert_intensity = surface_norm * pt_to_light_vec
                    if lambert_intensity > 0:
                        color += obj_base_color * obj.material.lambert * lambert_intensity

            # specular (reflective) light
            reflection_anchor = ray.direction.reflect(surface_norm).normalize()
            reflected_ray = Ray(intersection_pt, reflection_anchor)
            color += self._shoot_ray(reflected_ray, depth + 1) * obj.material.specular
            
            if(color.x >= 255): color.x = 255
            if(color.y >= 255): color.y = 255
            if(color.z >= 255): color.z = 255
                
            if(color.x <= 0): color.x = 0
            if(color.y <= 0): color.y = 0
            if(color.z <= 0): color.z = 0
            
            return color
        
        except Exception as e:
            if print_exception: print(str(e))
            return Color()

    def _get_intersection(self, ray):
        intersection = None
        for obj in self.objects:
            dist = obj.intersects(ray)
            if dist is not None and                 (intersection is None or dist < intersection[1]):
                intersection = obj, dist

        return intersection

class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        return math.sqrt(sum(num * num for num in self))

    def normalize(self, decimal=0):
        normal = self / self.norm()
        if(decimal == 0): return normal
        return Vector(*np.around(np.array(normal.to_list()), 2))

    def reflect(self, other):
        other = other.normalize()
        return self - 2 * (self * other) * other
    
    def to_list(self):
        return [self.x, self.y, self.z]

    def __str__(self):
        return "Vector({}, {}, {})".format(*self)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __pow__(self, exp):
        if exp != 2:
            raise ValueError("Exponent can only be two")
        else:
            return self * self

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

class Matrix:
    def __init__(self, *args):
        data = np.empty((0,3))
        
        for i in range(0, len(args), 3):
            data = np.vstack([data, args[i:i + 3]])
        
        self.data = data

    def __str__(self):
        return str(self.data)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                self.data[0][0] * other.x + self.data[0][1] * other.y + self.data[0][2] * other.z,
                self.data[1][0] * other.x + self.data[1][1] * other.y + self.data[1][2] * other.z,
                self.data[2][0] * other.x + self.data[2][1] * other.y + self.data[2][2] * other.z
            )
        else:
            return self.data
        
Point = Vector

class Color(Vector):
    def norm(self):
        return max(self.to_list())
    
    def __add__(self, other):
        return Color(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Color(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __truediv__(self, other):
        return Color(self.x / other, self.y / other, self.z / other)
    
    def __mul__(self, other):
        if isinstance(other, Color):
            return Color(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, Vector):
            raise TypeError('Unable to multiply Color by Vector')
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

class Sphere:
    """
    A sphere object.
    """

    def __init__(self, origin, radius, material, texture=None, bump_map=None):
        self.origin = origin
        self.radius = radius
        self.material = material
        self.texture = texture
        self.bump_map = bump_map

    def intersects(self, ray):
        sphere_to_ray = ray.origin - self.origin
        b = 2 * ray.direction * sphere_to_ray
        c = sphere_to_ray ** 2 - self.radius ** 2
        discriminant = b ** 2 - 4 * c

        if discriminant >= 0:
            dist = (-b - math.sqrt(discriminant)) / 2
            if dist > 0:
                return dist

    def surface_norm(self, pt):
        normal = (pt - self.origin).normalize()
        if(self.bump_map == None): return normal
        
        coords = CoordinateSystem(normal)
        A = coords.i
        B = coords.j
        
        u, v = self._to_unitary_square(pt)
        Bu, Bv = self.bump_map.from_unitary_square(u, v)
        
        D = A * Bu - B * Bv
        new_normal = (normal + D).normalize()
        return new_normal
    
    def _to_unitary_square(self, world_point):
        point = (world_point - self.origin) / self.radius
        
        u = 0.5 + (math.atan2(point.z, point.x) / (math.pi * 2))
        v = 0.5 - (2.0 * (math.asin(point.y) / (math.pi * 2)))
        
        return (u, v)
    
    def get_color(self, point=Point()):
        if(self.texture == None):
            return self.material.color
        
        norm_color = self.material.color.normalize()
        u, v = self._to_unitary_square(point)
        texel = self.texture.from_unitary_square(u, v)    
        material_contrib_texel = norm_color * texel
        
        return material_contrib_texel

class Material:

    def __init__(self, color, specular=0.5, lambert=1, ambient=0.2):
        self.color = color
        self.specular = specular
        self.lambert = lambert
        self.ambient = ambient
        
class Texture:
    
    def __init__(self, filepath=""):
        self.image = np.array(Image.open(filepath).convert('RGB'))
        h, w, color_channels = self.image.shape
        self.width = w - 1
        self.height = h - 1
    
    def from_unitary_square(self, u, v):
        posU = u * self.width
        posV = v * self.height
        
        if HasDecimalPlaces(posU) or HasDecimalPlaces(posV):
            posU = math.floor(posU)
            posV = math.floor(posV)
            
            startU = posU - 3 if posU == self.width else posU - 2 if posU > 0 else 0
            endU = posU + 3 if posU == 0 else posU + 2 if posU < self.width else self.width
            startV = posV - 3 if posV == self.height else posV - 2 if posV > 0 else 0
            endV = posV + 3 if posV == 0 else posV + 2 if posV < self.height else self.height
            
            texel = np.mean(self.image[startV:endV, startU:endU].flatten().reshape(16, 3), axis=0)
        else:
            texel = self.image[posV, posU]
            
        return Color(*texel)
        
class BumpMap(Texture):
    def __init__(self, filepath=""):
        self.image_obj = Image.open(filepath).convert('L')
        self.image = np.array(self.image_obj)
        h, w = self.image.shape
        self.width = w - 1
        self.height = h - 1
        
        KernelBu = (
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1
        )

        filterBu = ImageFilter.Kernel(
            size=(3, 3),
            kernel=KernelBu,
            scale=0.2,
            offset=0
        )

        KernelBv = (
             1,  1,  1,
             0,  0,  0,
            -1, -1, -1
        )

        filterBv = ImageFilter.Kernel(
            size=(3, 3),
            kernel=KernelBv,
            scale=0.2,
            offset=0
        )
        
        self.Bu = np.array(self.image_obj.filter(filterBu))
        self.Bv = np.array(self.image_obj.filter(filterBv))
        
    def from_unitary_square(self, u, v):
        posU = min(math.floor(u * self.width), self.width)
        posV = min(math.floor(v * self.height), self.height)
        
        return (self.Bu[posV, posU] / 255, self.Bv[posV, posU] / 255)
        
class Ray:        
    
    def __init__(self, origin, direction=None):
        if(direction):
            """
                Perspective ray
            """
            self.origin = origin
            self.direction = direction.normalize()
        else:
            """
                Parallel ray
            """
            z = origin.z - 1 if origin.z != 0 else origin.z - 2
            self.origin = Point(origin.x, origin.y, z)
            self.direction = Vector(0,0,z+1).normalize()
        
    def rotate_direction(self, x, y):
        matrix_x = Matrix(
            1,               0,                          0,
            0,    math.cos(math.radians(x)),  -math.sin(math.radians(x)),
            0,    math.sin(math.radians(x)),   math.cos(math.radians(x))
        )
        
        matrix_y = Matrix(
            math.cos(math.radians(y)),     0,   math.sin(math.radians(y)),
                      0,                   1,            0,
            -math.sin(math.radians(y)),    0,   math.cos(math.radians(y))
        )
        
        rot_direction = matrix_x * ( matrix_y * self.direction )
        self.direction = rot_direction.normalize()

    def point_at_dist(self, dist):
        return self.origin + self.direction * dist


# In[5]:


if __name__ == "__main__":
    
    earth = Sphere(Point(200, 200, 50), 40, Material(Color(255, 255, 255),
            lambert=0.6, specular=0.005), Texture('earth-day.jpg'), BumpMap('earth-day.jpg'))
    
    moon = Sphere(Point(80, 190, 75), 25, Material(Color(255, 255, 255),
            lambert=0.5, specular=0.3), Texture('moon-texture.png'), BumpMap('moon-craters.jpg'))
    
    sun = Sphere(Point(300, 200, 300), 80, Material(Color(255, 255, 255),
            lambert=0.6, specular=0.05), Texture('sun-texture.png'), BumpMap('sun-waves.jpg'))
    
    objects = [earth, moon, sun]
    lights = [
        Point(150, 200, 60), 
        Point(320, 230, 60), 
        Point(200, 50, 200), 
        Point(300, 200, 200), 
        Point(350, 250, 200), 
        Point(350, 50, 200)
    ]
    camera = Point(200, 200, 0)
    scene = Scene(camera, objects, lights, 640, 360, 1, 0, 0)
    pixels = scene.render()
    
    im = Image.fromarray(pixels.astype(np.uint8), "RGB")
    im.save("scene.png")

