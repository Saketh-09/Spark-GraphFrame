from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SocialNetworkAnalysis") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
    .getOrCreate()

# Load the dataset
file_path = "wiki-Vote.txt"
raw_data = spark.read.text(file_path)

# Filter out comment lines starting with '#'
edges = raw_data.filter(~col("value").startswith("#"))

# Split lines into FromNodeId and ToNodeId
edges = edges.selectExpr(
    "split(value, '\\t')[0] as src",
    "split(value, '\\t')[1] as dst"
)

# Convert data to integer types for consistency
edges = edges.select(col("src").cast("int"), col("dst").cast("int"))

# Extract unique vertices from edges
vertices = edges.selectExpr("src as id").union(edges.selectExpr("dst as id")).distinct()

# Create the GraphFrame
graph = GraphFrame(vertices, edges)

# a. Top 5 Nodes with Highest Outdegree
out_degree = graph.outDegrees.orderBy("outDegree", ascending=False).limit(5)
out_degree.show()
out_degree.write.csv("output/top5_outdegree.csv")

# b. Top 5 Nodes with Highest Indegree
in_degree = graph.inDegrees.orderBy("inDegree", ascending=False).limit(5)
in_degree.show()
in_degree.write.csv("output/top5_indegree.csv")

# c. PageRank
pagerank = graph.pageRank(resetProbability=0.15, maxIter=10)
top_pagerank = pagerank.vertices.orderBy("pagerank", ascending=False).limit(5)
top_pagerank.show()
top_pagerank.write.csv("output/top5_pagerank.csv")

# d. Connected Components
components = graph.connectedComponents()
largest_components = (
    components.groupBy("component")
    .count()
    .orderBy("count", ascending=False)
    .limit(5)
)
largest_components.show()
largest_components.write.csv("output/top5_components.csv")

# e. Triangle Counts
triangles = graph.triangleCount()
top_triangles = triangles.orderBy("count", ascending=False).limit(5)
top_triangles.show()
top_triangles.write.csv("output/top5_triangles.csv")

# Stop the Spark session
spark.stop()
