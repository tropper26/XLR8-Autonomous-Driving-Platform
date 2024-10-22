import networkx as nx


def compute_and_update_bounds(graphml_file):
    # Load the graph from a GraphML file
    G = nx.read_graphml(graphml_file)

    # Initialize variables to store min and max of x and y
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    # Compute the min and max values
    for node, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is not None and y is not None:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Update the graph with computed bounds
    G.graph["x_min"] = min_x
    G.graph["x_max"] = max_x
    G.graph["y_min"] = min_y
    G.graph["y_max"] = max_y

    # Write the graph back to a GraphML file
    nx.write_graphml(G, graphml_file)
    print("Graph updated and saved with bounds.")


if __name__ == "__main__":
    graphml_file_path = "../files/graphs/Competition_Track.graphml"

    compute_and_update_bounds(graphml_file_path)