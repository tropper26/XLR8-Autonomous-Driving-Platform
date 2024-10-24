import networkx as nx


def scale_positions(graph, scale_factor):
    for node in graph.nodes(data=True):
        if "x" in node[1] and "y" in node[1]:
            node[1]["x"] *= scale_factor
            node[1]["y"] *= scale_factor


def read_and_scale_graph(input_file, output_file, scale_factor):
    # Read the graph from the .graphml file
    graph = nx.read_graphml(input_file)

    # Scale the positions
    scale_positions(graph, scale_factor)

    # Write the graph to a new .graphml file
    nx.write_graphml(graph, output_file)
    print(f"Scaled graph written to {output_file}")


# Define the input and output files and the scale factor
input_file = "Test_Track.graphml"
output_file = "Test_Track1.graphml"
scale_factor = 10

# Run the function
read_and_scale_graph(input_file, output_file, scale_factor)