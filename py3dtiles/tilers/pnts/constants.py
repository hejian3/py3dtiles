# for all sort of size calculation, we consider a fake "thikness" of a point
# in most cases this is useless, but sometimes it's useful, for instance when
# calculating box sizes when all the points are coplanar
MIN_POINT_SIZE = 0.00001
