#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SAMPLES 10000
#define MAX_FEATURES 50
#define LINE_LEN 256

size_t samples = 0;
size_t features = 0;

double x[MAX_SAMPLES][MAX_FEATURES];
double x_norm[MAX_SAMPLES][MAX_FEATURES];
double y[MAX_SAMPLES];

double xmin[MAX_FEATURES];
double xmax[MAX_FEATURES];
double xrange[MAX_FEATURES];

int load_csv(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Cannot open file: %s\n", filename);
        return 0;
    }

    char line[LINE_LEN];
    samples = 0;

    while (fgets(line, LINE_LEN, fp) && samples < MAX_SAMPLES) {
        line[strcspn(line, "\r\n")] = 0;

        char *token = strtok(line, ",");
        size_t col = 0;

        while (token) {
            double val = atof(token);
            char *next = strtok(NULL, ",");

            if (!next)
                y[samples] = val;
            else
                x[samples][col++] = val;

            token = next;
        }

        if (samples == 0)
            features = col;

        samples++;
    }

    fclose(fp);
    printf("Loaded %zu samples with %zu features\n", samples, features);
    return 1;
}

void normalize_features(size_t train_count)
{
    for (size_t j = 0; j < features; j++) {
        xmin[j] = xmax[j] = x[0][j];

        for (size_t i = 1; i < train_count; i++) {
            if (x[i][j] < xmin[j]) xmin[j] = x[i][j];
            if (x[i][j] > xmax[j]) xmax[j] = x[i][j];
        }

        xrange[j] = xmax[j] - xmin[j];
        if (xrange[j] == 0.0) xrange[j] = 1.0;
    }

    for (size_t i = 0; i < samples; i++) {
        for (size_t j = 0; j < features; j++)
            x_norm[i][j] = (x[i][j] - xmin[j]) / xrange[j];
    }
}

double compute_gradients(size_t train_count,
                         double w[], double b,
                         double dw[], double *db)
{
    double loss = 0.0;

    for (size_t j = 0; j < features; j++)
        dw[j] = 0.0;
    *db = 0.0;

    for (size_t i = 0; i < train_count; i++) {
        double y_pred = b;
        for (size_t j = 0; j < features; j++)
            y_pred += w[j] * x_norm[i][j];

        double err = y_pred - y[i];
        loss += err * err;

        for (size_t j = 0; j < features; j++)
            dw[j] += 2.0 * err * x_norm[i][j];
        *db += 2.0 * err;
    }

    for (size_t j = 0; j < features; j++)
        dw[j] /= train_count;
    *db /= train_count;

    return loss / train_count;
}

int main(void)
{
    if (!load_csv("../data/student.csv"))
        return 1;

    size_t train_count = (size_t)(samples * 0.8);
    size_t test_count  = samples - train_count;

    normalize_features(train_count);

    double w[MAX_FEATURES] = {0};
    double b = 0.0;
    double lr = 0.1;

    for (size_t epoch = 0; epoch < 5000; epoch++) {
        double dw[MAX_FEATURES], db;
        double loss = compute_gradients(train_count, w, b, dw, &db);

        for (size_t j = 0; j < features; j++)
            w[j] -= lr * dw[j];
        b -= lr * db;

        if (epoch % 100 == 0)
            printf("Epoch %zu | Train MSE %.6f\n", epoch, loss);
    }

    for (size_t j = 0; j < features; j++)
        w[j] /= xrange[j];

    double shift = 0.0;
    for (size_t j = 0; j < features; j++)
        shift += w[j] * xmin[j];
    b -= shift;

    double test_loss = 0.0;
    for (size_t i = train_count; i < samples; i++) {
        double y_pred = b;
        for (size_t j = 0; j < features; j++)
            y_pred += w[j] * x[i][j];
        double err = y_pred - y[i];
        test_loss += err * err;
    }
    test_loss /= test_count;

    printf("\nTest MSE: %.6f\n", test_loss);
    printf("\nFinal model (original feature space):\n");

    for (size_t j = 0; j < features; j++)
        printf("Weight %zu: %.6f\n", j, w[j]);
    printf("Bias: %.6f\n", b);

    return 0;
}
