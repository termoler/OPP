#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include "mpi.h"
#define N 1000

double getNorma(const std::vector<double>& value) {
    double norma = 0;
    for(double i : value) norma += i * i;
    return norma;
}

std::vector<double> differenceVectors(std::vector<double>& x1, std::vector<double>& y1){
    int sizeX1 = (int)x1.size();
    for(int i = 0; i < sizeX1 ; i++) x1[i] -= y1[i];
    return x1;
}

bool conditionStop(const std::vector<double>& xNext, const std::vector<double>& b){
    double epsilon = 0.00001;
    double normaB = 0;
    double normaPartOfB = getNorma(b);
    MPI_Allreduce(&normaPartOfB, &normaB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double normaX = 0;
    double normaPartOfX = getNorma(xNext);
    MPI_Allreduce(&normaPartOfX, &normaX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return normaPartOfX < normaPartOfB * epsilon * epsilon;
}

std::vector<double> getPartOfDecision(const std::vector<double>& mat, std::vector<double>& partOfX,
                                      std::vector<double>& partOfB, int rank, int size, const std::vector<int>& gatheredNumbers) {
    int sizeMatrix = (int) (mat.size() / N);
    std::vector<double> partXNext(sizeMatrix, 0);
    std::vector<double> xNext(N, 0);
    bool flag = false;
    double tay = 1.0 / (N);
    std::vector<int> displs;
    displs.push_back(0);
    for (int i = 1; i < size; i++) displs.push_back(displs[i - 1] + gatheredNumbers[i - 1]);
    int counterIterations = 0;

    while (true) {
        int idx = (rank + counterIterations)%size;
        if(counterIterations == 0){
            for(size_t i = 0; i < sizeMatrix; i++){
                for(size_t j = 0, k = 0; j < gatheredNumbers[idx]; j++, k++){//тк sizeMatrix == count(количество элементов после деления)
                    partXNext[i] += mat[i * N + j + displs[idx]] * partOfX[j];
                }
            }
        }
        else if (counterIterations > 0 && counterIterations < size) {
            std::vector<double> tmpVector(gatheredNumbers[idx], rank);
            MPI_Request req[2];
            MPI_Status st[2];
            MPI_Isend(&partOfX[0], (int)partOfX.size(), MPI_DOUBLE, (rank - counterIterations + size) % size, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&tmpVector[0], (int)tmpVector.size(), MPI_DOUBLE, (rank + counterIterations) % size, 0, MPI_COMM_WORLD, &req[1]);
            MPI_Waitall(2, req, st);
            for(size_t i = 0; i < sizeMatrix; i++){
                for(size_t j = 0, k = 0; j < gatheredNumbers[idx]; j++, k++){//тк sizeMatrix == count(количество элементов после деления)
                    partXNext[i] += mat[i * N + j + displs[idx]] * tmpVector[j];
                }
            }
        }
        else {
            partXNext = differenceVectors(partXNext, partOfB);
            flag = conditionStop(partXNext, partOfB);

            MPI_Bcast(&flag, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

            if(flag) break;
            for (size_t i = 0; i < sizeMatrix; i++) {
                partXNext[i] = partOfX[i] - tay * partXNext[i];
            }
            MPI_Allgatherv(&partXNext[0], (int)partXNext.size(), MPI_DOUBLE, &xNext[0], &gatheredNumbers[0], &displs[0],
                           MPI_DOUBLE, MPI_COMM_WORLD);
            partOfX = partXNext;

            counterIterations = 0;
            continue;
        }
        counterIterations++;
    }
    return xNext;
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
    auto startTime = MPI_Wtime();

    for(size_t i = 0; i < N; i++) if(rank == i % size) count++;
    mat.resize(N * count);
    std::fill(mat.begin(), mat.end(), 1);
    MPI_Allgather(&count, 1, MPI_INT, gatheredNumbers.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<double> partOfX(count, 0);
    std::vector<double> partOfB(count, N + 1);

    int sum = 0;
    for(size_t j = 0; j < rank; j ++){
        sum += gatheredNumbers[j];
    }
    for(size_t i = 0; i < count; i++){
        mat[i * N + counterTwo + sum] = 2;
        counterTwo++;
    }
    std::vector<double> partOfDecision = getPartOfDecision(mat, partOfX, partOfB, rank, size, gatheredNumbers);
    auto endTime = MPI_Wtime();
    if(rank == 0){
//        for (int i = 0; i < partOfDecision.size(); i++) {
//            std::cout << partOfDecision[i] << "  ";
//            if ((i + 1) % N == 0) {
//                std::cout << "    " <<rank << "\n";
//            }
//        }
        std::cout << "\n" << endTime - startTime;
        std::cout << "\n";
    }
    MPI_Finalize();
    return 0;
}
