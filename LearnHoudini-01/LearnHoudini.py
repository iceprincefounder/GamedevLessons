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

geo.addAttrib(hou.attribType.Vertex, "N", hou.Vector3(0, 0, 0)) #初始化法线参数
geo.addAttrib(hou.attribType.Vertex, "uv", hou.Vector3(0, 0, 0)) #初始化UV参数
for point_indices in poly_point_indices:
    #为当前geo创建Polygon
    poly = geo.createPolygon()
    list_index = poly_point_indices.index(point_indices) #获取当前循环数
    for point_index in point_indices:
        #为当前geo创建Vertex
        point = points[point_index] #通过Index找到对应的点
        vertex = poly.addVertex(point) #创建Vertex
        if list_index%2 == 0: #和2取余
            normal = hou.Vector3(0, 0, 1) #正方向法线
        else:
            normal = hou.Vector3(0, 0, -1) #反方向法线
        #设置法线属性
        vertex.setAttribValue("N", normal)
        #设置UV属性
        uv = hou.Vector3(point.position()[0], point.position()[1], 0)
        vertex.setAttribValue("uv", uv)