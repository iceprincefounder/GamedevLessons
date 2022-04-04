node = hou.pwd() #获取当前节点
geo = node.geometry() #获取当前节点的geo

#创建顶点位置
point_positions = ((0,0,0), (1,0,0), (1,1,0), (0,1,0))
#创建顶点位置索引
poly_point_indices = ((0,1,2), (2,3,0))

points = []
for position in point_positions:
    #为当前geo创建Point
    point = geo.createPoint()
    point.setPosition(position)
    points.append(point)

for point_indices in poly_point_indices:
    #为当前geo创建Polygon
    poly = geo.createPolygon()
    for point_index in point_indices:
        #为当前geo创建Vertex
        point = points[point_index] #通过Index找到对应的点
        poly.addVertex(point) #创建Vertex