#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <math.h>

#include <mpi.h>

#include "Utility.hpp"

#define DEBUG 0


// New generate problem function for mpi splitting
// Each process only generates its own points, others are skipped via random.discard
// Query points are put at the front of the resulting array
// Function is kinda convoluted due to taking care of remainders
float* generate_problem(int seed, int dim, int local_num_points, int points_n, int total_points, int num_queries){
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-100, 100);
    float* x = (float*)calloc(dim * (local_num_points + num_queries), sizeof(float));

    random.discard(points_n*dim);
    // generate points for the process
    for(int n = num_queries; n < local_num_points + num_queries; ++n){
        for(int d = 0; d < dim; ++d){
            *(x + n * dim + d) = distribution(random);
        }
    }

    random.discard((total_points - points_n - local_num_points) * dim);
    // generate query points
    for(int n = 0; n < num_queries; ++n){
        for(int d = 0; d < dim; ++d){
            *(x + n * dim + d) = distribution(random);
        }
    }
    
    return x;
}

/***************************************************************************************/
float Point::distance_squared(Point &a, Point &b){
    if(a.dimension != b.dimension){
        std::cout << "Dimensions do not match!" << std::endl;
        exit(1);
    }
    float dist = 0;
    for(int i = 0; i < a.dimension; ++i){
        float tmp = a.coordinates[i] - b.coordinates[i];
        dist += tmp * tmp;
    }
    return dist;
}
/***************************************************************************************/


/***************************************************************************************/
Node* build_tree_rec(Point** point_list, int num_points, int depth){
    if (num_points <= 0){
        return nullptr;
    }

    if (num_points == 1){
        return new Node(point_list[0], nullptr, nullptr); 
    }

    int dim = point_list[0]->dimension;

    // sort list of points based on axis
    int axis = depth % dim;
    using std::placeholders::_1;
    using std::placeholders::_2;

    std::sort(
        point_list, point_list + (num_points - 1), 
        std::bind(Point::compare, _1, _2, axis));

    // select median
    Point** median = point_list + (num_points / 2);
    Point** left_points = point_list;
    Point** right_points = median + 1;

    int num_points_left = num_points / 2;
    int num_points_right = num_points - (num_points / 2) - 1; 

    // left subtree
    Node* left_node = build_tree_rec(left_points, num_points_left, depth + 1);
    
    // right subtree
    Node* right_node = build_tree_rec(right_points, num_points_right, depth + 1);

    // return median node
    return new Node(*median, left_node, right_node); 
}

Node* build_tree(Point** point_list, int num_nodes){
    return build_tree_rec(point_list, num_nodes, 0);
}
/***************************************************************************************/


/***************************************************************************************/
Node* nearest(Node* root, Point* query, int depth, Node* best, float &best_dist) {
    // leaf node
    if (root == nullptr){
        return nullptr; 
    }

    int dim = query->dimension;
    int axis = depth % dim;

    Node* best_local = best;
    float best_dist_local = best_dist;
    
    float d_euclidian = root->point->distance_squared(*query);
    float d_axis = query->coordinates[axis] - root->point->coordinates[axis];
    float d_axis_squared = d_axis * d_axis;

    if (d_euclidian < best_dist_local){
        best_local = root;
        best_dist_local = d_euclidian;
    }

    Node* visit_branch;
    Node* other_branch;

    if(d_axis < 0){
        // query point is smaller than root node in axis dimension, i.e. go left
        visit_branch = root->left;
        other_branch = root->right;
    } else{
        // query point is larger than root node in axis dimension, i.e. go right
        visit_branch = root->right;
        other_branch = root->left;
    }

    Node* further = nearest(visit_branch, query, depth + 1, best_local, best_dist_local);
    if (further != nullptr){
        float dist_further = further->point->distance_squared(*query);
        if (dist_further < best_dist_local){
            best_dist_local = dist_further;
            best_local = further;
        }
    }
    
    if (d_axis_squared < best_dist_local) {
        further = nearest(other_branch, query, depth + 1, best_local, best_dist_local);
        if (further != nullptr){
            float dist_further = further->point->distance_squared(*query);
            if (dist_further < best_dist_local){
                // best_dist_local = dist_further;
                best_local = further;
            }
        }
    }
    
    return best_local;
}


Node* nearest_neighbor(Node* root, Point* query){
    float best_dist = root->point->distance_squared(*query);
    return nearest(root, query, 0, root, best_dist);
}


/***************************************************************************************/
int main(int argc, char **argv){
    int seed = 0;
    int dim = 0;
    int num_points = 0;
    int num_queries = 10;
    int data[3];

    MPI_Init(&argc, &argv);
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Datatype msg;
    MPI_Type_contiguous(3, MPI_INT, &msg);
    MPI_Type_commit(&msg);

    if(comm_rank == 0){
        #if DEBUG
            // for measuring your local runtime
            auto tick = std::chrono::high_resolution_clock::now();
            Utility::specify_problem(argc, argv, &seed, &dim, &num_points);
        #else
            Utility::specify_problem(&seed, &dim, &num_points);
        #endif
        data[0] = seed;
        data[1] = dim;
        data[2] = num_points;

    }

    MPI_Bcast(&data,1,msg,0,MPI_COMM_WORLD);
    seed = data[0];
    dim = data[1];
    num_points = data[2];
    
    // we could divide our number of points by the number of mpi process
    // so each process deal with their own local_points and build local_tree
    // in the end we compare the results of the nearest point out of each local_tree
    // Finally, choose the nearest one.
    int local_num_points = num_points / comm_size;

    int reminder = num_points % comm_size;
    int points_n = local_num_points*comm_rank;

    if(comm_rank == comm_size - 1)
    {   
        local_num_points += reminder;
    }

    float* x = generate_problem(seed, dim, local_num_points, points_n, num_points, num_queries);

    Point** local_points = (Point**)calloc(local_num_points, sizeof(Point*));

    for(int n = 0; n < local_num_points; ++n){
        local_points[n] = new Point(dim, n + points_n + 1, x + (n + num_queries) * dim);
    }

    // storing local_min_dist for each query
    float* local_min_dists = (float*)calloc(num_queries, sizeof(float));

    // build tree
    Node* local_tree = build_tree(local_points, local_num_points);


    // for each query, find nearest neighbor
    for(int q = 0; q < num_queries; ++q){
        float* x_query = x + q * dim;   // query points are now in the beginning of x

        Point query(dim, num_points + q, x_query);

        Node* local_res = nearest_neighbor(local_tree, &query);
        
        // output min-distance (i.e. to query point)
        local_min_dists[q] = query.distance(*local_res->point);
    }

    #if DEBUG
        // in case you want to have further debug information about
        // the query point and the nearest neighbor
        // std::cout << "Query: " << query << std::endl;
        // std::cout << "NN: " << *res->point << std::endl << std::endl;
    #endif

    float* global_min_dists = (float*)calloc(num_queries, sizeof(float));
    MPI_Reduce(local_min_dists, global_min_dists, num_queries, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

    if(comm_rank == 0){

        for(auto i = 0; i < num_queries; i++)
        {
            float min_distance = global_min_dists[i];
            Utility::print_result_line(num_points+i, min_distance);
        }

        #if DEBUG
            // for measuring your local runtime
            auto tock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = tock - tick;
            std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
        #endif

        std::cout << "DONE" << std::endl;

    }

    // clean-up
    Utility::free_tree(local_tree);

    for(int n = 0; n < local_num_points; ++n){
        delete local_points[n];
    }

    free(local_points);
    free(x);
    free(local_min_dists);
    free(global_min_dists);


    (void)argc;
    (void)argv;
    MPI_Finalize();
    return 0;
}
