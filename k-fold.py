#!/usr/bin/env python
# coding: utf-8

# In[9]:


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

point=[3,5,4,6,7,3]
smooth_curve(point)


# In[ ]:




