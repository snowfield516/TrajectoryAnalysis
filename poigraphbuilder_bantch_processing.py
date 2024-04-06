import geopandas as gpd
import duckdb
import osmnx as ox
import networkx as nx
from  shapely.geometry import Point
import matplotlib.pyplot as plt
import logging
import pickle
from rtree import index
from shapely.wkt import loads
from pyproj import CRS, Transformer
class POIGraphBuilder:
    def __init__(self):
        # define a MultiDiGraph
        self.G = nx.MultiDiGraph()
        # define a node dictionary to check the duplicated node point
        self.node_dict = {}
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.conn = duckdb.connect("C://Users//snowf//Desktop//Umassdata//breadcrumbs_test.duckdb")
        self.conn.install_extension("spatial")
        self.conn.load_extension("spatial")

    def convert_poi_to_graph(self,user_id,distance=120):
        query_locations = f"SELECT id, fulltime, poi_id, ST_AsText(poigeom), descriptiontext,groupid FROM breadcrumbs_test.main.user_{user_id}_joinpoi_{distance}m_new_distinct"
        user_location = self.conn.execute(query_locations).fetchall()
        # 初始化一个有向图来存储POI图数据结构
        self.G = nx.MultiDiGraph()
        # 上一个访问的POI ID，初始化为None
        last_visited_poi_id = None

        for location in user_location:
            id, fulltime, poi_id,poi_geom,descriptiontext,groupid = location

            poi_position = loads(poi_geom)
            # 在POI数据中找到最近的POI
            poi_x, poi_y = (poi_position.x, poi_position.y) # Project coordinates
            #logging.info("The POI: %s || poi_longitude: %s || and poi_latitude: %s and distance : %s ", poi_id, poi_position.x, poi_position.y, descriptiontext)
            # logging.info("The POI: %s || poi_longitude: %s || and poi_latitude: %s and distance : %s ", poi_id, poi_x, poi_y, descriptiontext)

            poi_XY = (poi_x, poi_y)
            if poi_id not in self.G:
                self.G.add_node(poi_id, x=poi_x, y=poi_y, pos=poi_XY, label=poi_id, nodeid=poi_id,groupid=groupid, poi=descriptiontext)
            #if closest_poi and closest_poi[0] not in visited_poi_ids:
            #visited_poi_ids.add(closest_poi[0])
            if last_visited_poi_id is not None and not self.G.has_edge(last_visited_poi_id, poi_id):
                # add count
                # count++
            #if last_visited_poi_id is not None:

                #self.G.add_edge(last_visited_poi_id,poi_id)

                self.G.add_edge(last_visited_poi_id, poi_id,  startnode=last_visited_poi_id, endnode=poi_id)
            last_visited_poi_id = poi_id
        #print("user_id"+str(user_id))
        self.save_graph_to_file(f"./result/poigraph_{user_id}.gml")
        #print(self.G)
        # pos = nx.get_node_attributes(self.G,'pos')  # 确定节点的布局
        # nx.draw_networkx_nodes(self.G, pos)
        # nx.draw_networkx_edges(self.G, pos)
        # nx.draw_networkx_labels(self.G, pos)
        # plt.ion()
        #plt.show(block=True)
        #shortest_path = nx.shortest_path(self.G, source=120, target=3)
        #logging.info("The shortest path between start and end nodes are : %s ", shortest_path)

    def convert_poi_to_graph_new(self, user_id,distance=120):
        query_locations = f"SELECT id, fulltime, poi_id, ST_AsText(poigeom), descriptiontext, groupid FROM breadcrumbs_test.main.user_{user_id}_joinpoi_{distance}m_new_distinct"
        user_location = self.conn.execute(query_locations).fetchall()
        self.G = nx.MultiDiGraph()
        last_visited_poi_id = None
        userkey = 'u' + str(user_id)
        for location in user_location:
            id, fulltime, poi_id, poi_geom, descriptiontext, groupid = location
            poi_position = loads(poi_geom)
            poi_x, poi_y = (poi_position.x, poi_position.y)

            poi_XY = (poi_x, poi_y)
            if poi_id not in self.G:
                self.G.add_node(poi_id, x=poi_x, y=poi_y, pos=poi_XY, label=poi_id, nodeid=poi_id, groupid=groupid,
                                poi=descriptiontext, user_ids={userkey: 1})
            else:
                # 如果POI节点已经存在，更新它的user_ids属性
                if userkey in self.G.nodes[poi_id]['user_ids']:
                    self.G.nodes[poi_id]['user_ids'][userkey] += 1
                else:
                    self.G.nodes[poi_id]['user_ids'][userkey] = 1

            if last_visited_poi_id is not None:
                if not self.G.has_edge(last_visited_poi_id, poi_id):
                    self.G.add_edge(last_visited_poi_id, poi_id, startnode=last_visited_poi_id, endnode=poi_id,
                                    user_counts={userkey: 1})
                else:
                    # 如果边已存在，更新user_counts
                    if userkey in self.G[last_visited_poi_id][poi_id][0]['user_counts']:
                        self.G[last_visited_poi_id][poi_id][0]['user_counts'][userkey] += 1
                    else:
                        self.G[last_visited_poi_id][poi_id][0]['user_counts'][userkey] = 1

            last_visited_poi_id = poi_id

        self.save_graph_to_file(self.G,f"./result/poigraph_{user_id}.gml")

    def merge_graphs(self, graphs):
        merged_graph = nx.MultiDiGraph()
        usercount_dict = {}  # 辅助字典，记录节点的usercount

        for graph in graphs:
            # 合并节点
            for node, data in graph.nodes(data=True):
                if node not in merged_graph:
                    merged_graph.add_node(node, **data)
                    usercount_dict[node] = set(data['user_ids'].keys())
                else:
                    # 合并user_ids
                    existing_user_ids = merged_graph.nodes[node]['user_ids']
                    for user_id, count in data['user_ids'].items():
                        userkey = 'u'+ str(user_id)
                        if userkey in existing_user_ids:
                            existing_user_ids[userkey] += count
                        else:
                            existing_user_ids[userkey] = count
                    usercount_dict[node].update(data['user_ids'].keys())
                # 更新usercount属性
                merged_graph.nodes[node]['usercount'] = len(usercount_dict[node])

            # 合并边
            for u, v, key, data in graph.edges(keys=True, data=True):
                if merged_graph.has_edge(u, v, key):
                    existing_user_counts = merged_graph[u][v][key]['user_counts']
                    # 如果边已经存在，则合并user_counts
                    for user_id, count in data['user_counts'].items():
                        userkey = 'u' + str(user_id)
                        if userkey in merged_graph[u][v][key]['user_counts']:
                            existing_user_counts[user_id] += count
                        else:
                            existing_user_counts[user_id] = count
                else:
                    # 添加新边
                    merged_graph.add_edge(u, v, key=key, **data)
        self.save_graph_to_file(merged_graph,f"./result/poigraph_merged.gml")



    def save_graph_to_file(self,graph, saveurl):
        nx.write_gml(graph, saveurl)
        # save to binary data using pickle
        #with open('../result/graph1.pkl', 'wb') as f:
        #    pickle.dump(self.G, f)
        # print the graph data
        logging.info(graph)

    def calculate_shortest_path(self, startnode, endnode):
        print("-----------call function----------")
        shortest_path = nx.shortest_path(self.G, source=startnode, target=endnode)
        logging.info("The shortest path between start and end nodes are : %s ", shortest_path)

    def build_nodes_index(self):
        # Create a new R-tree index and save it to a file
        p = index.Property()
        p.dat_extension = 'dat'
        p.idx_extension = 'idx'
        p.ifx_extension = 'ifx'
        p.filename = 'rtree_index'
        idx = index.Index('./data/rtree_index', properties=p)
        print("------create a index")
        # Add nodes to the index
        for n, d in self.G.nodes(data=True):
            # Assume 'coord' is a tuple (x, y)
            pos = d['pos']
            idx.insert(n, pos + pos)  # (x, y, x, y) is the bounding box

        # The index is automatically saved to 'rtree_index.dat'

        # Later, you can load the index from the file
        #idx = index.Index('rtree_index')

    def load_saved_graph_by_user(self, user_id):
        G = nx.read_gml(f'./result/poigraph_{user_id}.gml')

        # logging.info(duplicate_edges)

    def load_graphs_for_users(self, user_ids):
        graphs = []
        for user_id in user_ids:
            file_path = f"./result/poigraph_{user_id}.gml"
            try:
                graph = nx.read_gml(file_path)
                graphs.append(graph)
            except FileNotFoundError:
                print(f"Graph file for user {user_id} not found.")
        return graphs

    def load_saved_graph(self,user_id):
        G = nx.read_gml(f'./result/poigraph_{user_id}.gml')
        logging.info(self.G)
        duplicate_edges = self.check_duplicate_edges()
        logging.info(duplicate_edges)

    def load_saved_binary_graph(self):
        #load the binary graph data
        with open('./result/graph1.pkl', 'rb') as f:
            self.G = pickle.load(f)
            logging.info(self.G)

    def check_saved_index(self):
        idx = index.Index('./data/rtree_index')
        print(idx)


    def check_duplicate_edges(self):
        # Get all edges in the graph
        edges = list(self.G.edges(data=True))
        # Create a set to store unique edges
        edge_set = set()

        # List to store duplicate edges
        duplicate_edges = []

        # Check each edge
        for edge in edges:
            # Get the nodes of the edge (without considering the direction)
            edge_nodes =(edge[0], edge[1])

            # If the edge has been seen before, it's a duplicate
            if edge_nodes in edge_set:
                duplicate_edges.append(edge)
            else:
                edge_set.add(edge_nodes)

        return duplicate_edges

    ### first step: create user location by minute
    def create_user_minute_table(self, user_id):
        query = f"""
        CREATE TABLE user_{user_id}_minute AS (
        WITH RankedData AS (
            SELECT *,  
            RANK() OVER (PARTITION BY timestr ORDER BY fulltime ASC) as rank
            FROM location_projected WHERE user_id = {user_id}
        )
        SELECT *
        FROM RankedData
        WHERE rank = 1);
        """
        self.conn.execute(query)
    ### second step: create user location group by 20 minutes time window;
    def create_user_20_timewindow_table(self, user_id):
        query = f"""
        create table user_{user_id}_minute_by_window_center as (
        WITH global_min_max AS (
            SELECT 
                MIN(timestamp) AS min_fulltime, 
                MAX(timestamp) AS max_fulltime
            FROM user_{user_id}_minute
        ),
        time_windows AS (
            SELECT
                user_{user_id}_minute.*,
                FLOOR((timestamp - (SELECT min_fulltime FROM global_min_max))/(20 * 60)) AS window_id 
            FROM user_{user_id}_minute, global_min_max
        ),
        averaged AS (
            SELECT
                window_id,
                ST_Centroid(ST_Collect(list(geom))) AS geom,
                FIRST(day) As day,
                FIRST(hour) As hour,
                FIRST(minute) As minute,
                FIRST(fulltime) As fulltime
            FROM time_windows
            GROUP BY window_id
        )
        SELECT
            row_number() OVER (ORDER BY window_id) AS id,
            day,hour,minute,fulltime,geom,
            ST_X(geom) AS center_lon,
            ST_Y(geom) AS center_lat
        FROM averaged
        )
        """
        self.conn.execute(query)
    ### third step: extract user's poi table
    def create_user_poi_table(self, user_id):
        query = f"""
        CREATE TABLE poi_{user_id} AS (
            SELECT * FROM poi_clustered WHERE user_id = {user_id}
        );
        """
        self.conn.execute(query)

    ### fourth step get the location with the poi data:
    def generate_user_poi_join_table(self, user_id, distance=120):
        query = f"""
        CREATE TABLE user_{user_id}_joinpoi_{distance}m AS (
        WITH UserPOIDistances AS (
        SELECT
            a.id,
            a.day,
            a.hour,
            a.minute,
            a.fulltime,
            a.geom as locationgeom,
            b.cluster_id AS poi_id,
            b.geom as poigeom,
            b.descript_2 as descriptiontext,
            ST_Distance(locationgeom, poigeom) AS poidistance,
            ROW_NUMBER() OVER (PARTITION BY a.id ORDER BY poidistance ASC) AS rn
        FROM
            (SELECT ROW_NUMBER() OVER (ORDER BY fulltime ASC) AS id, * FROM main.user_{user_id}_minute_by_window_center) a, main.poi_{user_id} b
        WHERE
            ST_DWithin(locationgeom, poigeom, {distance})
        )
        SELECT
            id,
            day,
            hour,
            minute,
            fulltime,
            locationgeom,
            poi_id,
            poidistance,
            poigeom,
            descriptiontext,
            rn
        FROM
            UserPOIDistances
        WHERE
            rn = 1
        ORDER BY
            id
        );
        """
        self.conn.execute(query)

    ###third step get the location with the poi data:
    def generate_user_poi_join_table_distinct(self, user_id, distance=120):
        query = f"""
        create table user_{user_id}_joinpoi_{distance}m_new_distinct as (
        WITH ValueChanges AS (
          SELECT
            id,
            fulltime,
            poi_id,          
            poigeom,
            descriptiontext,
            CASE
              WHEN LAG(poi_id) OVER (ORDER BY id) = poi_id THEN 0
              ELSE 1
            END AS isNewGroup
          FROM
            user_{user_id}_joinpoi_{distance}m
        ),
        Groups AS (
          SELECT
            id,
            fulltime,
            poi_id,
            poigeom,
            descriptiontext,
            CAST(SUM(isNewGroup) OVER (ORDER BY id) AS BIGINT) AS groupId           
          FROM
            ValueChanges
        ),
        Grouped as (
        SELECT
            id,
            fulltime,
            poi_id,
            poigeom,
            descriptiontext,
            groupId,
          ROW_NUMBER() OVER (PARTITION BY groupId ORDER BY id) AS groupOrder
        FROM
          Groups       
          )
        SELECT
          id,
          fulltime,
          poi_id,
          poigeom,
          descriptiontext,
          groupId
        FROM
          Grouped
        WHERE
          groupOrder = 1
        ORDER BY
          id, fulltime
          );
        """
        self.conn.execute(query)

    def get_all_users(self):
        query = "SELECT distinct(user_id) FROM location_projected order by user_id"
        results = self.conn.execute(query).fetchall()
        #print(results)
        # Extract user_ids from tuples and return them as a list
        user_ids = [result[0] for result in results]

        return user_ids

    def drop_all_tables(self):
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND  table_name not in ('location_projected','poi_projected','poi_clustered'); "

        tables_df = self.conn.execute(query).fetchdf()
        # 逐个删除表
        for table_name in tables_df['table_name']:
            drop_table_query = f"DROP TABLE IF EXISTS {table_name}"
            self.conn.execute(drop_table_query)



if __name__ == '__main__':
    model = POIGraphBuilder()
    #清空除了'location_projected','poi_projected','poi_clustered'的其他的表
    # model.drop_all_tables()
    #Stage1：处理每个User的location信息，构造Graph数据
    #80个数据处理2分钟。

    # user_ids = model.get_all_users()
    # for user_id in user_ids:
    #     ##Assuming these methods are defined to handle a single user_id
    #     model.create_user_minute_table(user_id)
    #     model.create_user_20_timewindow_table(user_id)
    #     model.create_user_poi_table(user_id)
    #     model.generate_user_poi_join_table(user_id)
    #     model.generate_user_poi_join_table_distinct(user_id)
    #     model.convert_poi_to_graph_new(user_id)

    #Stage2：读取每个User的Graph数据，并合并成为一个大的Graph
    user_ids = model.get_all_users()
    graphs = model.load_graphs_for_users(user_ids)
    model.merge_graphs(graphs)

    # user_id = 102
    # # 1.创建用户表(按照分钟提取）
    # model.create_user_minute_table(user_id)
    # # 2.创建用户表(按照20分钟窗口提取)
    # model.create_user_20_timewindow_table(user_id)
    # # 3.创建用户POI表
    # model.create_user_poi_table(user_id)
    # # 4.根据距离对location和POI表进行关联，提取关联上的数据
    # model.generate_user_poi_join_table(user_id)
    # # 5.按照点位的出行顺序进行排序，合并连续相同的POI，相同POI点集只保留唯一的POI
    # model.generate_user_poi_join_table_distinct(user_id)
    # model.convert_poi_to_graph(user_id)

