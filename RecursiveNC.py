import os
from tkinter.tix import CheckList

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
from statistics import mean
from shapely.geometry import shape
import datetime
import multiprocessing
from shapely import speedups, Point, Polygon, LineString,ops, MultiLineString, box
import pandas as pd
import rasterio
from rasterio import features
import htb, htb2, htb3
from pyproj import CRS
import csv
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely import polygonize
import matplotlib.pyplot as plt
import networkx as nx
import powerlaw
from mgwr.gwr import GWR

starttime = datetime.datetime.now()

def mergecsvfiles(file_dir, output_file):
    # 列出路径下全部文件名
    csv_files = [file for file in os.listdir(file_dir) if file.endswith('.csv')]
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(file_dir, file),encoding = "GBK", engine='python', error_bad_lines=False)
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    columns_to_remove = ['address','adname','page_publish_time','adcode','pname','_id','type']
    merged_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    merged_df.to_csv(output_file, index=False, encoding="GBK")

def csv2shp(csv_file, shp_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [r for r in reader]
    df = pd.DataFrame(data, columns=headers)
    df['lng'] = df['location'].apply(lambda x: x.split('，')[0])
    print(df['lng'])
    df['lat'] = df['location'].apply(lambda x: x.split('，')[1])
    df['geometry'] = df.apply(lambda x: Point((float(x.lng), float(x.lat))), axis=1)
    df = df.drop(['lng', 'lat', 'location', 'name', 'Unnamed: 10'], axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = 'EPSG:4326'
    gdf.to_file(shp_file)

def mergeshps(shp1, shp2, outputfile):
    gdf1 = gpd.read_file(shp1)
    gdf2 = gpd.read_file(shp2)
    merged_gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))
    merged_gdf['z'] = 0
    if 'cityname' in merged_gdf.columns:
        merged_gdf = merged_gdf.drop(columns=['cityname'])
    merged_gdf = merged_gdf.to_crs(epsg=3857)
    merged_gdf.to_file(outputfile)
    print('Merge shp files successfully!')

def NCGen_test():
    ptshp = gpd.read_file(r'C:\zhgren\ChinaPOI\POI_shp\ChinaPOI_beijing.shp')
    inputcrs = ptshp.crs
    print('input number:'+str(len(ptshp)))
    pts = ptshp['geometry'].apply(lambda geom: (geom.x, geom.y)).values
    ptslist = []
    for pt in pts:
        xylist = list(pt)
        ptslist.append(xylist)
    ptarray = np.array(ptslist)
    tri = Delaunay(ptarray)
    tript = tri.points
    triedge = tri.simplices
    coor_groups = [tript[x] for x in triedge]
    trilines = []
    for i in range(0, len(coor_groups)):
        triline1 = LineString([coor_groups[i][0], coor_groups[i][1]])
        trilines.append(triline1)
        triline2 = LineString([coor_groups[i][1], coor_groups[i][2]])
        trilines.append(triline2)
        triline3 = LineString([coor_groups[i][2], coor_groups[i][0]])
        trilines.append(triline3)
    # remove duplicates
    linegdf = gpd.GeoDataFrame(geometry=trilines)
    linegdf['geometry'] = linegdf.normalize()
    linegdf_single = linegdf.drop_duplicates('geometry')
    print('edge number:'+str(len(linegdf_single)))
    meanedgelength = mean([x.length for x in linegdf_single['geometry']])
    print('mean length:'+str(meanedgelength))
    # meanedgelength = 208
    shorterlinesdf = linegdf_single[linegdf_single['geometry'].length < meanedgelength]
    print('shorter edge:'+str(len(shorterlinesdf)))
    shorterlinelist = shorterlinesdf['geometry'].values
    tinpolycollections = polygonize(shorterlinelist)
    tinpolylist = []
    for geom in tinpolycollections.geoms:
        tinpolylist.append(geom)
    print('polygonized:'+str(len(tinpolylist)))
    tinpolydf = gpd.GeoDataFrame(geometry=tinpolylist)
    tinpolydf = tinpolydf.set_crs(inputcrs)
    # tinpolydf.to_file(r'C:\zhgren\ChinaPOI\POI_shp\ChinaPOI_region1hk_tin.shp')
    # tinpolydf.plot()
    # plt.show()
    dissolved = ops.unary_union(tinpolydf['geometry'])
    print('nc number:'+str(len(dissolved.geoms)))
    nc1list = []
    for geom in dissolved.geoms:
        nc1list.append(geom)
    nc = gpd.GeoDataFrame(geometry=nc1list)
    nc = nc.set_crs(inputcrs)
    nc.to_file(r'C:\zhgren\ChinaPOI\POI_shp\ChinaPOI_beijing_nc.shp')
    print('nc generated successfully!')

    # # nc.plot(color='red')
    # # plt.show()
    # return ptshp, nc

def NCGen(ptdf,polync):
    print('input number:'+str(len(ptdf)))
    inputcrs = ptdf.crs
    pts = ptdf['geometry'].apply(lambda geom: (geom.x, geom.y)).values
    ptslist = []
    for pt in pts:
        xylist = list(pt)
        ptslist.append(xylist)
    ptarray = np.array(ptslist)
    tri = Delaunay(ptarray)
    tript = tri.points
    triedge = tri.simplices
    coor_groups = [tript[x] for x in triedge]
    trilines = []
    for i in range(0, len(coor_groups)):
        triline1 = LineString([coor_groups[i][0], coor_groups[i][1]])
        trilines.append(triline1)
        triline2 = LineString([coor_groups[i][1], coor_groups[i][2]])
        trilines.append(triline2)
        triline3 = LineString([coor_groups[i][2], coor_groups[i][0]])
        trilines.append(triline3)
    # remove duplicates
    linegdf = gpd.GeoDataFrame(geometry=trilines)
    linegdf['geometry'] = linegdf.normalize()
    linegdf_single = linegdf.drop_duplicates('geometry')

    # Important from the second recursion! select lines within the polygon
    # linegdf_single = linegdf_single[linegdf_single['geometry'].within(polync)]

    meanedgelength = mean([x.length for x in linegdf_single['geometry']])
    shorterlinesdf = linegdf_single[linegdf_single['geometry'].length < meanedgelength]
    ratio = len(shorterlinesdf)/len(linegdf_single)
    print('ratio: '+str(ratio))

    # there are far more small edges than long edges
    if (ratio > 0.6):
        shorterlinelist = shorterlinesdf['geometry'].values
        tinpolycollections = polygonize(shorterlinelist)
        tinpolylist = []
        for geom in tinpolycollections.geoms:
            tinpolylist.append(geom)
        tinpolydf = gpd.GeoDataFrame(geometry=tinpolylist)
        dissolved = ops.unary_union(tinpolydf['geometry'])
        nc1list = []
        if dissolved.geom_type == 'Polygon':
            nc1list.append(dissolved)
        else:
            for geom in dissolved.geoms:
                nc1list.append(geom)
        nc = gpd.GeoDataFrame(geometry=nc1list)
        nc = nc.set_crs(inputcrs)
        return nc
    else:
        nc = []
        return nc

def NCGen_start(inputpt):
    ptshp = gpd.read_file(inputpt)
    inputcrs = ptshp.crs
    print('input number:'+str(len(ptshp)))
    pts = ptshp['geometry'].apply(lambda geom: (geom.x, geom.y)).values
    ptslist = []
    for pt in pts:
        xylist = list(pt)
        ptslist.append(xylist)
    ptarray = np.array(ptslist)
    tri = Delaunay(ptarray)
    tript = tri.points
    triedge = tri.simplices
    coor_groups = [tript[x] for x in triedge]
    trilines = []
    for i in range(0, len(coor_groups)):
        triline1 = LineString([coor_groups[i][0], coor_groups[i][1]])
        trilines.append(triline1)
        triline2 = LineString([coor_groups[i][1], coor_groups[i][2]])
        trilines.append(triline2)
        triline3 = LineString([coor_groups[i][2], coor_groups[i][0]])
        trilines.append(triline3)
    # remove duplicates
    linegdf = gpd.GeoDataFrame(geometry=trilines)
    linegdf['geometry'] = linegdf.normalize()
    linegdf_single = linegdf.drop_duplicates('geometry')
    print('edge number:'+str(len(linegdf_single)))
    meanedgelength = mean([x.length for x in linegdf_single['geometry']])
    print('mean length:'+str(meanedgelength))
    shorterlinesdf = linegdf_single[linegdf_single['geometry'].length < meanedgelength]
    print('shorter edge:'+str(len(shorterlinesdf)))
    shorterlinelist = shorterlinesdf['geometry'].values
    tinpolycollections = polygonize(shorterlinelist)
    tinpolylist = []
    for geom in tinpolycollections.geoms:
        tinpolylist.append(geom)
    print('polygonized:'+str(len(tinpolylist)))
    tinpolydf = gpd.GeoDataFrame(geometry=tinpolylist)
    # tinpolydf.plot()
    # plt.show()
    dissolved = ops.unary_union(tinpolydf['geometry'])
    print('nc number:'+str(len(dissolved.geoms)))
    nc1list = []
    for geom in dissolved.geoms:
        nc1list.append(geom)
    nc = gpd.GeoDataFrame(geometry=nc1list)
    nc = nc.set_crs(inputcrs)
    # nc.plot(color='red')
    # plt.show()
    return ptshp, nc

def ncjoin(ptdf, ncdf):
    joined = gpd.sjoin(ptdf, ncdf, how='left', predicate='intersects')
    grouped = joined.groupby('index_right').size().rename('count').reset_index()
    ncjoined = ncdf.merge(grouped, left_index=True, right_on='index_right', how='left')
    H = len(htb2.htb(ncjoined['count'].tolist()))
    return ncjoined, H

def recurNC(ncjoined_df, ptdf):
    print('start recursive decomposition...')
    cutoff = mean([c for c in ncjoined_df['count']])
    print(cutoff)
    nclist = ncjoined_df[ncjoined_df['count'] > cutoff]
    resultlist = []
    Hlist = []
    ptnumber = []

    print(len(nclist))
    for i in range(0, len(nclist)):
        p = nclist['geometry'].iloc[i]
        subset = ptdf[ptdf['geometry'].intersects(p)]
        # print(len(subset))
        ptnumber.append(len(subset))
        nc = NCGen(subset, p)
        print(len(nc))
        if len(nc) > 10:
            ncjoined, h = ncjoin(ptdf, nc)
            if h>=3:
                print('H: ' + str(h))
                resultlist.append(ncjoined)
                Hlist.append(h)
    print('length of Decomposable: ' + str(len(resultlist)))
    return resultlist, Hlist, ptnumber


def LivingGen():
    print('input pt and initial nc first')
    # generate starting NCs from POI data
    inputpt = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan.shp'
    ptdf, ncdf = NCGen_start(inputpt)
    ncjoined, h1 = ncjoin(ptdf, ncdf)
    print('H1: ' + str(h1))
    # ncjoined.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NCjoin.shp', driver="ESRI Shapefile")

    # First level recursive decomposition
    # result1, hlist1, ptn = recurNC(ncjoined, ptdf)
    # ncdf1 = pd.concat(result1)
    # ncdf1.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R1.shp', driver="ESRI Shapefile")
    #
    # result2, hlist2, ptn2 = recurNC(ncdf1, ptdf)
    # ncdf2 = pd.concat(result2)
    # ncdf2.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R2.shp', driver="ESRI Shapefile")
    #
    # result3, hlist3, ptn3 = recurNC(ncdf2, ptdf)
    # ncdf3 = pd.concat(result3)
    # ncdf3.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R3.shp', driver="ESRI Shapefile")
    #
    # result4, hlist4, ptn4 = recurNC(ncdf3, ptdf)
    # ncdf4 = pd.concat(result4)
    # ncdf4.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R4.shp.shp', driver="ESRI Shapefile")

    # result5, hlist5, ptn5 = recurNC(ncdf4, ptdf)
    # ncdf5 = pd.concat(result5)
    # ncdf5.to_file(r'C:\zhgren\ChinaPOI\POI_shp\ChinaPOI_xian_NC_R5.shp.shp', driver="ESRI Shapefile")

def LivingCal():
    inputsub0 = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NCjoin.shp'
    sub0 = gpd.read_file(inputsub0)
    inputsub1 = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R1.shp'
    sub1 = gpd.read_file(inputsub1)
    inputsub2 = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R2.shp'
    sub2 = gpd.read_file(inputsub2)
    inputsub3 = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_NC_R3.shp'
    sub3 = gpd.read_file(inputsub3)

    L0 = len(sub0) * len(htb2.htb(sub0['count'].tolist()))
    print ('L0: ' + str(L0))
    Dlist = []
    Lrlist = []
    slist = []
    hlist = []
    for i in range(0, len(sub0)):
        p = sub0['geometry'].iloc[i]
        subset = sub1[sub1['geometry'].intersects(p)]
        if len(subset) > 1:
            Dlist.append(subset)
    print (len(Dlist))
    for i in range(0, len(Dlist)):
        h = len(htb2.htb(Dlist[i]['count'].tolist()))
        hlist.append(h)
        s = len(Dlist[i])
        slist.append(s)
        li = s * h
        Lrlist.append(li)
    df = pd.DataFrame({'s': slist, 'h': hlist, 'Lr': Lrlist})
    df.to_csv(r'C:\zhgren\ChinaPOI\ChinaTop10\Livingness\ChinaPOI_wuhan_NC_R1.csv', index=False)
    L1 = sum(Lrlist)
    print ('L1: ' + str(L1))

    Dlist = []
    Lrlist = []
    slist = []
    hlist = []
    for i in range(0, len(sub1)):
        p = sub1['geometry'].iloc[i]
        subset = sub2[sub2['geometry'].intersects(p)]
        if len(subset) > 1:
            Dlist.append(subset)
    print (len(Dlist))
    for i in range(0, len(Dlist)):
        h = len(htb2.htb(Dlist[i]['count'].tolist()))
        hlist.append(h)
        s = len(Dlist[i])
        slist.append(s)
        li = s * h
        Lrlist.append(li)
    df = pd.DataFrame({'s': slist, 'h': hlist, 'Lr': Lrlist})
    df.to_csv(r'C:\zhgren\ChinaPOI\ChinaTop10\Livingness\ChinaPOI_wuhan_NC_R2.csv', index=False)
    L2 = sum(Lrlist)
    print('L2: ' + str(L2))

    Dlist = []
    Lrlist = []
    slist = []
    hlist = []
    for i in range(0, len(sub2)):
        p = sub2['geometry'].iloc[i]
        subset = sub3[sub3['geometry'].intersects(p)]
        if len(subset) > 1:
            Dlist.append(subset)
    print(len(Dlist))
    for i in range(0, len(Dlist)):
        h = len(htb2.htb(Dlist[i]['count'].tolist()))
        hlist.append(h)
        s = len(Dlist[i])
        slist.append(s)
        li = s * h
        Lrlist.append(li)
    df = pd.DataFrame({'s': slist, 'h': hlist, 'Lr': Lrlist})
    df.to_csv(r'C:\zhgren\ChinaPOI\ChinaTop10\Livingness\ChinaPOI_wuhan_NC_R3.csv', index=False)
    L3 = sum(Lrlist)
    print('L3: ' + str(L3))


def graphviz():
    G = nx.balanced_tree(3, 5)
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
    plt.axis("equal")
    plt.show()

def createTree():
    subshp = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_Dall.shp'
    sub = gpd.read_file(subshp)
    nodelist = []
    nodelabel = []
    nodelevel = []
    fatherlist = []
    childlist = []

    for i in range(0, len(sub)):
        nodelist.append(sub['uid'].iloc[i]+1)
        nodelabel.append(i+1)
        nodelevel.append(sub['level'].iloc[i]+2)
    nodelist.append(100000)
    nodelabel.append(100000)
    nodelevel.append(1)
    nodedf = pd.DataFrame({'Id': nodelist, 'Label': nodelabel, 'level': nodelevel})
    nodedf.to_csv(r'C:\zhgren\ChinaPOI\ChinaTop10\Graph\ChinaPOI_wuhanD_Node.csv', index=False)

    fathersubs = sub[sub['level'] == 0]
    print(len(fathersubs))
    childsubs = sub[sub['level'] == 1]
    print(len(childsubs))

    for i in range(0, len(fathersubs)):
        fatherid = fathersubs['uid'].iloc[i]
        fatherlist.append(100000)
        childlist.append(fatherid + 1)
    print(len(fatherlist))

    print('start creating tree level1-2')
    for i in range(0, len(fathersubs)):
        father = fathersubs['geometry'].iloc[i]
        child = childsubs[childsubs['geometry'].intersects(father)]
        fatherid = fathersubs['uid'].iloc[i]
        childid = child['uid'].values
        for j in range(0, len(childid)):
            fatherlist.append(fatherid+1)
            childlist.append(childid[j]+1)


    print('start creating tree level2-3')
    fathersubs = sub[sub['level'] == 1]
    print(len(fathersubs))
    childsubs = sub[sub['level'] == 2]
    print(len(childsubs))
    for i in range(0, len(fathersubs)):
        father = fathersubs['geometry'].iloc[i]
        child = childsubs[childsubs['geometry'].intersects(father)]
        fatherid = fathersubs['uid'].iloc[i]
        childid = child['uid'].values
        for j in range(0, len(childid)):
            fatherlist.append(fatherid+1)
            childlist.append(childid[j]+1)

    # print('start creating tree level3-4')
    # fathersubs = sub[sub['level'] == 2]
    # print(len(fathersubs))
    # childsubs = sub[sub['level'] == 3]
    # print(len(childsubs))
    # for i in range(0, len(fathersubs)):
    #     father = fathersubs['geometry'].iloc[i]
    #     child = childsubs[childsubs['geometry'].intersects(father)]
    #     fatherid = fathersubs['uid'].iloc[i]
    #     childid = child['uid'].values
    #     for j in range(0, len(childid)):
    #         fatherlist.append(fatherid + 1)
    #         childlist.append(childid[j] + 1)

    edgedf = pd.DataFrame({'Source': fatherlist, 'Target': childlist})
    edgedf.to_csv(r'C:\zhgren\ChinaPOI\ChinaTop10\Graph\ChinaPOI_wuhanD_Edge.csv', index=False)
    print('tree created successfully!')

def extactDecomp():
    subshp = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_SubsH1.shp'
    sub = gpd.read_file(subshp)
    crs = sub.crs
    fathersubs = sub[sub['level'] == 0]
    print(len(fathersubs))
    childsubs1 = sub[sub['level'] == 1]
    print(len(childsubs1))
    childsubs2 = sub[sub['level'] == 2]
    print(len(childsubs2))
    childsubs3 = sub[sub['level'] == 3]
    print(len(childsubs3))

    Dlist = []
    for i in range(0, len(fathersubs)):
        father = fathersubs['geometry'].iloc[i]
        child = childsubs1[childsubs1['geometry'].within(father)]
        if len(child) > 1:
            Dlist.append(fathersubs.iloc[i])
    dshp = gpd.GeoDataFrame(Dlist)
    dshp.set_crs(crs)
    dshp.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_D0.shp')
    print('D0 created successfully!')

    Dlist = []
    for i in range(0, len(childsubs1)):
        father = childsubs1['geometry'].iloc[i]
        child = childsubs2[childsubs2['geometry'].within(father)]
        if len(child) > 1:
            Dlist.append(childsubs1.iloc[i])
    dshp = gpd.GeoDataFrame(Dlist)
    dshp.set_crs(crs)
    dshp.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_D1.shp')
    print('D1 created successfully!')
    Dlist = []
    for i in range(0, len(childsubs2)):
        father = childsubs2['geometry'].iloc[i]
        child = sub[sub['geometry'].within(father)]
        if len(child) > 1:
            Dlist.append(childsubs2.iloc[i])
    dshp = gpd.GeoDataFrame(Dlist)
    dshp.set_crs(crs)
    dshp.to_file(r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_wuhan_D2.shp')
    print('D2 created successfully!')

def fitpowerlaw():
    shppath = r'C:\zhgren\ChinaPOI\ChinaTop10\ChinaPOI_beijing_Subs.shp'
    shapefile = gpd.read_file(shppath)
    # based on area
    arealist = shapefile['geometry'].area.tolist()
    arealist.sort(reverse=True)
    print(len(arealist))
    # based on count
    countlist = shapefile['count'].tolist()
    countlist.sort(reverse=True)
    print(len(countlist))

    # Fit power-law distribution using maximum likelihood estimation
    fit = powerlaw.Fit(countlist, discrete=True, estimate_discrete=True)
    R, p = fit.distribution_compare('power_law', 'lognormal')
    # Print estimated parameters
    print("Alpha (exponent of the power law):", fit.alpha)
    print("Xmin (minimum value for power-law behavior):", fit.xmin)
    print("Sigma (Standard error for power-law behavior):", fit.sigma)
    print("R: ", R)
    print("p: ", p)
    # Plot the data and power-law fit
    fig = fit.plot_ccdf(color='b', linewidth=0, marker='o', markersize=5)
    fit.power_law.plot_ccdf(color='g', linestyle='--', ax=fig)
    plt.xlabel('X')
    plt.ylabel('P(X>x)')
    plt.title('Power-law Distribution Fit')
    plt.tight_layout()
    plt.show()

    x_values, y_values = fit.ccdf()
    df = pd.DataFrame({'x': x_values/1000000, 'y': y_values})
    df.to_csv(r'C:\zhgren\ChinaPOI\Powerlaw\ChinaPOI_beijingRall_ccdf.csv', index=False, header=False)


if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    print(cpus)
    speedups.enabled
    print(speedups.enabled)
    print(starttime)

    # dirname = r'C:\zhgren\ChinaPOI\POI_csv_merge'
    # outputfile = r'C:\zhgren\ChinaPOI\2018-POICSV-6.csv'
    # mergecsvfiles(dirname, outputfile)

    # csv_file = r'C:\zhgren\ChinaPOI\POI_csv_merge\2018-POICSV-6.csv'
    # shp_file = r'C:\zhgren\ChinaPOI\POI_shp\2018-POI-6.shp'
    # csv2shp(csv_file, shp_file)

    # NCGen()
    # LivingGen()
    # graphviz()
    # LivingCal()
    # createTree()
    # extactDecomp()
    fitpowerlaw()
    print(datetime.datetime.now())