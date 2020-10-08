#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
from PIL import Image


# In[2]:


def HasDecimalPlaces(number):
    return (number / 2) % 1 > 0


# In[3]:


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
    
    return Point(
        center.x + (point.x - half_w), 
        center.y + (point.y - half_h),
        center.z + distance
    ).to_list()
        
class Scene:
    """
    The scene that gets rendered. Contains information like the camera
    position, the different objects present, etc.
    """

    def __init__(self, camera, objects, lights, width, height, viewport_dist):
        self.camera = camera
        self.objects = objects
        self.lights = lights
        self.width = width
        self.height = height
        self.viewport_dist = viewport_dist

    def render(self):
        """
        Return a `self.height`x`self.width` 2D array of `Color`s representing
        the color of each pixel, obtained via ray-tracing.
        """
        
        pixels = np.zeros((self.height, self.width, 3))
        grid = GeneratePixelGrid(16)
        
        for y in range(self.height):
            for x in range(self.width):
                noisy_points = grid + ToViewportPoint(self.camera, Point(x, y), self.width, self.height, self.viewport_dist)
                noisy_ray_directions = noisy_points - self.camera.to_list()
#                 check what is causing perspective error
                colors = np.zeros((0, 3))
                
                for ray_direction in noisy_ray_directions:
                    direction = Vector(*ray_direction.tolist())
                    ray = Ray(self.camera, direction)
                    color = self._shoot_ray(ray)
                    colors = np.vstack([colors, color.to_list()])
                
                pixels[y, x, :] = np.mean(colors, axis=0)

        return pixels

    def _shoot_ray(self, ray, depth=0, max_depth=5):
        try:
            color = Color()

            if depth >= max_depth:
                return color

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
            return color
        
        except:
            return Color()

    def _get_intersection(self, ray):
        """
        If ray intersects any of `self.objects`, return `obj, dist` (the object
        itself, and the distance to it). Otherwise, return `None`.
        """

        intersection = None
        for obj in self.objects:
            dist = obj.intersects(ray)
            if dist is not None and                 (intersection is None or dist < intersection[1]):
                intersection = obj, dist

        return intersection

class Vector:
    """
    A generic 3-element vector. All of the methods should be self-explanatory.
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        return math.sqrt(sum(num * num for num in self))

    def normalize(self):
        return self / self.norm()

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

    def __init__(self, origin, radius, material, texture=None):
        self.origin = origin
        self.radius = radius
        self.material = material
        self.texture = texture

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
        return (pt - self.origin).normalize()
    
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
        w, h, color_channels = self.image.shape
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
            
            texel = np.mean(self.image[startU:endU, startV:endV].flatten().reshape(16, 3), axis=0)
        else:
            texel = self.image[posU, posV]
            
        return Color(*texel)
        
        
class Ray:
    """
    A mathematical ray.
    """

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

    def point_at_dist(self, dist):
        return self.origin + self.direction * dist


# In[4]:


if __name__ == "__main__":
    objects = [
#         Sphere(
#             Point(150, 120, -20), 80, Material(Color(255, 0, 0),
#             specular=0.2)),
#         Sphere(
#             Point(420, 120, 0), 100, Material(Color(0, 0, 255),
#             specular=0.8)),
#         Sphere(Point(320, 240, -40), 50, Material(Color(0, 255, 0))),
#         Sphere(
#             Point(300, 200, 200), 100, Material(Color(255, 255, 0),
#             specular=0.8)),
#         Sphere(Point(300, 130, 100), 40, Material(Color(255, 0, 255))),
        Sphere(Point(200, 200, 55), 40, Material(Color(255, 255, 255),
            lambert=0.5, specular=0.15), Texture('earth-texture.jpg')),
        Sphere(Point(120, 190, 65), 25, Material(Color(255, 255, 255),
            lambert=0.5, specular=0.3), Texture('moon-texture.png')),
        ]
    lights = [Point(200, 100, 50), Point(300, 300, 60), Point(-300, 300, 60), Point(110, 180, 35)]
    camera = Point(200, 200, 0)
    scene = Scene(camera, objects, lights, 600, 400, 70)
    pixels = scene.render()
    
    im = Image.fromarray(pixels.astype(np.uint8), "RGB")
    im.save("scene.png")
#     im.show()

