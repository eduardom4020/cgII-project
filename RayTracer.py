#!/usr/bin/env python
# coding: utf-8

# In[41]:


import math
import random
import numpy as np
from PIL import Image


# In[51]:


def InnerPixelRandomPoints(point, rays_amt):
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

    return grid + point.to_list()
        
class Scene:
    """
    The scene that gets rendered. Contains information like the camera
    position, the different objects present, etc.
    """

    def __init__(self, camera, objects, lights, width, height):
        self.camera = camera
        self.objects = objects
        self.lights = lights
        self.width = width
        self.height = height

    def render(self):
        """
        Return a `self.height`x`self.width` 2D array of `Color`s representing
        the color of each pixel, obtained via ray-tracing.
        """
        
        pixels = np.zeros((self.height, self.width, 3))

        for y in range(self.height):
            for x in range(self.width):
                noisy_points = InnerPixelRandomPoints(Point(x, y), 16)
                noisy_ray_directions = noisy_points - self.camera.to_list()
                
                colors = np.zeros((0, 3))
                
                for ray_direction in noisy_ray_directions:
                    direction = Vector(*ray_direction.tolist())
                    ray = Ray(self.camera, direction)
                    color = self._shoot_ray(ray)
                    colors = np.vstack([colors, color.to_list()])
                
                pixels[y, x, :] = np.mean(colors, axis=0)

        return pixels

    def _shoot_ray(self, ray, depth=0, max_depth=5):
        """
        Recursively trace a ray through the scene, returning the color it
        accumulates.
        """

        color = Color()

        if depth >= max_depth:
            return color

        intersection = self._get_intersection(ray)
        if intersection is None:
            return color

        obj, dist = intersection
        intersection_pt = ray.point_at_dist(dist)
        surface_norm = obj.surface_norm(intersection_pt)

        # ambient light
        color += obj.material.color * obj.material.ambient

        # lambert shading
        for light in self.lights:
            pt_to_light_vec = (light - intersection_pt).normalize()
            pt_to_light_ray = Ray(intersection_pt, pt_to_light_vec)
            if self._get_intersection(pt_to_light_ray) is None:
                lambert_intensity = surface_norm * pt_to_light_vec
                if lambert_intensity > 0:
                    color += obj.material.color * obj.material.lambert *                         lambert_intensity

        # specular (reflective) light
        reflected_ray = Ray(
            intersection_pt, ray.direction.reflect(surface_norm).normalize())
        color += self._shoot_ray(reflected_ray, depth + 1) *             obj.material.specular
        return color

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


# In[52]:


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
            return self.x * other.x + self.y * other.y + self.z * other.z;
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

# Since 3D points and RGB colors are effectively 3-element vectors, we simply
# declare them as aliases to the `Vector` class to take advantage of all its
# defined operations (like overloaded addition, multiplication, etc.) while
# improving readability (so we can use `color = Color(0xFF)` instead of
# `color = Vector(0xFF)`).
Point = Vector
Color = Vector

class Sphere:
    """
    A sphere object.
    """

    def __init__(self, origin, radius, material):
        self.origin = origin
        self.radius = radius
        self.material = material

    def intersects(self, ray):
        """
        If `ray` intersects sphere, return the distance at which it does;
        otherwise, `None`.
        """

        sphere_to_ray = ray.origin - self.origin
        b = 2 * ray.direction * sphere_to_ray
        c = sphere_to_ray ** 2 - self.radius ** 2
        discriminant = b ** 2 - 4 * c

        if discriminant >= 0:
            dist = (-b - math.sqrt(discriminant)) / 2
            if dist > 0:
                return dist

    def surface_norm(self, pt):
        """
        Return the surface normal to the sphere at `pt`.
        """

        return (pt - self.origin).normalize()

class Material:

    def __init__(self, color, specular=0.5, lambert=1, ambient=0.2):
        self.color = color
        self.specular = specular
        self.lambert = lambert
        self.ambient = ambient

class Ray:
    """
    A mathematical ray.
    """

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

    def point_at_dist(self, dist):
        return self.origin + self.direction * dist

if __name__ == "__main__":
    objects = [
        Sphere(
            Point(150, 120, -20), 80, Material(Color(0xFF, 0, 0),
            specular=0.2)),
        Sphere(
            Point(420, 120, 0), 100, Material(Color(0, 0, 0xFF),
            specular=0.8)),
        Sphere(Point(320, 240, -40), 50, Material(Color(0, 0xFF, 0))),
        Sphere(
            Point(300, 200, 200), 100, Material(Color(0xFF, 0xFF, 0),
            specular=0.8)),
        Sphere(Point(300, 130, 100), 40, Material(Color(0xFF, 0, 0xFF))),
        Sphere(Point(300, 1000, 0), 700, Material(Color(0xFF, 0xFF, 0xFF),
            lambert=0.5)),
        ]
    lights = [Point(200, -100, 0), Point(600, 200, -200)]
    camera = Point(200, 200, -400)
    scene = Scene(camera, objects, lights, 600, 400)
    pixels = scene.render()
    
    im = Image.fromarray(pixels.astype(np.uint8), "RGB")
#     im.save("scene.png")
    im.show()

