#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_VERTICES 10000

// Graph structure
struct Graph {
    int V;
    int** adj;
};

void initGraph(struct Graph* g, int vertices) {
    g->V = vertices;
    g->adj = (int**)malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        g->adj[i] = (int*)calloc(vertices, sizeof(int));
    }
}

void freeGraph(struct Graph* g) {
    for (int i = 0; i < g->V; i++) {
        free(g->adj[i]);
    }
    free(g->adj);
}

void addEdge(struct Graph* g, int u, int v) {
    g->adj[u][v] = 1;
    g->adj[v][u] = 1;
}

void generateRandomEdges(struct Graph* g, int max_edges) {
    int edges = 0;
    while (edges < max_edges) {
        int u = rand() % g->V;
        int v = rand() % g->V;
        if (u != v && g->adj[u][v] == 0) {
            addEdge(g, u, v);
            edges++;
        }
    }
}

int sequentialBFS(struct Graph* g, int start) {
    int V = g->V;
    int* visited = (int*)calloc(V, sizeof(int));
    int* q = (int*)malloc(MAX_VERTICES * sizeof(int));
    int front = 0, rear = 0;
    int count = 0;

    visited[start] = 1;
    q[rear++] = start;

    printf("BFS (Sequential): ");

    while (front != rear) {
        int node = q[front++];
        printf("%d ", node);
        count++;

        for (int i = 0; i < V; i++) {
            if (g->adj[node][i] && !visited[i]) {
                visited[i] = 1;
                q[rear++] = i;
            }
        }
    }

    printf("\nVisited %d nodes (Sequential)\n", count);
    free(visited);
    free(q);
    return count;
}

int parallelBFS(struct Graph* g, int start) {
    int V = g->V;
    int* visited = (int*)calloc(V, sizeof(int));
    int* q = (int*)malloc(MAX_VERTICES * sizeof(int));
    int front = 0, rear = 0;
    int count = 0;

    visited[start] = 1;
    q[rear++] = start;

    printf("BFS (Parallel): ");

    while (front != rear) {
        int localFront = front;
        int localRear = rear;

        #pragma omp parallel
        {
            int* thread_queue = (int*)malloc(MAX_VERTICES * sizeof(int));
            int thread_rear = 0;

            #pragma omp for
            for (int i = localFront; i < localRear; i++) {
                int node = q[i];

                #pragma omp critical
                {
                    printf("%d ", node);
                    count++;
                }

                for (int j = 0; j < V; j++) {
                    if (g->adj[node][j]) {
                        // Atomic compare-and-swap to avoid duplicate enqueue
                        if (__sync_bool_compare_and_swap(&visited[j], 0, 1)) {
                            thread_queue[thread_rear++] = j;
                        }
                    }
                }
            }

            // Merge thread-local queues into the global queue
            #pragma omp critical
            {
                for (int i = 0; i < thread_rear; i++) {
                    q[rear++] = thread_queue[i];
                }
            }

            free(thread_queue);
        }

        front = localRear;
    }

    printf("\nVisited %d nodes (Parallel)\n", count);
    free(visited);
    free(q);
    return count;
}

int main() {
    int i;
    double inputs[5][6];

    for (i = 0; i < 5; i++) {  // Loop 5 times to handle 5 graphs
        printf("Enter number of vertices for graph %d (1 to 10000): ", i + 1);
        scanf("%lf", &inputs[i][0]);

        if (inputs[i][0] < 1 || inputs[i][0] > 10000) {
            printf("Invalid vertex count.\n");
            return 1;
        }

        int V = (int)inputs[i][0];
        int max_edges = V * (V - 1) / 4;  // You can adjust this as needed
        printf("Generating up to %d edges...\n", max_edges);

        struct Graph g;
        initGraph(&g, V);
        srand(time(NULL) + i);  // Seed random numbers with a different value each time
        generateRandomEdges(&g, max_edges);

        int start;
        printf("Enter starting vertex for BFS (0 to %d): ", V - 1);
        scanf("%d", &start);

        if (start < 0 || start >= V) {
            printf("Invalid start vertex.\n");
            return 1;
        }

        clock_t seq_start = clock();
        int seq_count = sequentialBFS(&g, start);
        clock_t seq_end = clock();
        inputs[i][3] = ((double)(seq_end - seq_start)) / CLOCKS_PER_SEC;

        double par_start = omp_get_wtime();
        int par_count = parallelBFS(&g, start);
        double par_end = omp_get_wtime();
        inputs[i][4] = par_end - par_start;

        inputs[i][5] = (inputs[i][4] > 0) ? (inputs[i][3] / inputs[i][4]) : 0;

        if (seq_count != par_count) {
            printf("⚠️ Mismatch in node counts: sequential = %d, parallel = %d\n", seq_count, par_count);
        }

        freeGraph(&g);
    }

    printf("\n---------------------------------------------------------------\n");
    printf("| Graph | Vertices | Sequential Time | Parallel Time | Speedup |\n");
    printf("---------------------------------------------------------------\n");

    for (i = 0; i < 5; i++) {
        printf("| %5d | %8.0lf | %15.6f | %13.6f | %7.2f |\n",
               i + 1, inputs[i][0], inputs[i][3], inputs[i][4], inputs[i][5]);
    }
    printf("---------------------------------------------------------------\n");

    return 0;
}
