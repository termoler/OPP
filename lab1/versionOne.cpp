#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include "mpi.h"

#define N 1000

double getNorma(std::vector<double>& value) {
    double norma = 0;
    for(double i : value) norma += i * i;
    return norma;
}

std::vector<double> mulMatrixOnVector(const std::vector<double>& matrix, std::vector<double>& y1){
    int sizeMatrix = (int)(matrix.size()/N);
    std::vector<double> x1y1(sizeMatrix, 0);
    for(size_t i = 0; i < sizeMatrix; i++){
        for(size_t j = 0; j < N; j++){
            x1y1[i] += matrix[i * N + j] * y1[j];
        }
    }
    return x1y1;
}
std::vector<double> differenceVectors(std::vector<double>& x1, std::vector<double>& y1){
    int sizeX1 = (int)x1.size();
    for(int i = 0; i < sizeX1 ; i++) x1[i] -= y1[i];
    return x1;
}

bool conditionStop(std::vector<double>& xNext, std::vector<double>& b){
    double epsilon = 0.00001;
    double epsilonOpt = epsilon * epsilon * getNorma(b);
    return getNorma(xNext) < epsilonOpt;
}

std::vector<double> getDecision(const std::vector<double>& matrix, std::vector<double>& x,
                                std::vector<double>& b, int size, int& sum, const std::vector<int>& gatheredNumbers) {
    int sizeMatrix = (int)(matrix.size()/N);
    std::vector<double> partXNext(sizeMatrix, 0);
    std::vector<double> xNext(N, 0);
    bool flag = false;
    double tay = 0.001;
    std::vector<int> displs;
    displs.push_back(0);
    for(int i = 1; i < size; i++){
        displs.push_back(displs[i - 1] + gatheredNumbers[i - 1]);
    }

    while (true) {
        partXNext = mulMatrixOnVector(matrix, x);

        partXNext = differenceVectors(partXNext, b);

        flag = conditionStop(partXNext, b);

        MPI_Bcast(&flag, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if(flag) break;
        for (size_t i = 0; i < sizeMatrix; i++) {
            partXNext[i] = x[i + sum] - tay * partXNext[i];
        }
        MPI_Allgatherv(&partXNext[0], sizeMatrix, MPI_DOUBLE, &xNext[0] , &gatheredNumbers[0], &displs[0] ,MPI_DOUBLE ,MPI_COMM_WORLD);
        x = xNext;
    }

    return x;
}

int main() {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<double> mat;
    int count = 0;
    int counterTwo = 0;
    std::vector<int> gatheredNumbers(size);
//    if(size > N) return 0;
    auto startTime = MPI_Wtime();

    for(size_t i = 0; i < N; i++) if(rank == i % size) count++;
    mat.resize(N * count);
    std::fill(mat.begin(), mat.end(), 1);
    MPI_Allgather(&count, 1, MPI_INT, gatheredNumbers.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int sum = 0;
    for(size_t j = 0; j < rank; j ++){
        sum += gatheredNumbers[j];
    }
    for(size_t i = 0; i < count; i++){
        mat[i * N + counterTwo + sum] = 2;
        counterTwo++;
    }
    std::vector<double> b(N, N + 1);
    std::vector<double> x(N, 0);

    std::vector<double> decision = getDecision(mat, x, b, size,  sum, gatheredNumbers);
    auto endTime = MPI_Wtime();

    if(rank == 0){
//        for (int i = 0; i < decision.size(); i++) {
//            std::cout << decision[i]  << "  ";
//            if ((i + 1) % N == 0) {
//                std::cout << "    " <<rank << "\n";
//            }
//        }
        std::cout << "\n" << endTime - startTime;

    }

    MPI_Finalize();
    return 0;
}
