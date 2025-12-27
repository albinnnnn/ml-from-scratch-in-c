#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_SAMPLES 1000
#define LINE_LEN 256

double data[MAX_SAMPLES][2];      // raw x, y
double data_norm[MAX_SAMPLES][2]; // normalized x (train), raw y

double xmin, xmax;
size_t sample_count = 0;

/* ---------------- CSV loading ---------------- */
int load_csv(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to open file: %s\n", filename);
        return 0;
    }

    char header[LINE_LEN];
    fgets(header, LINE_LEN, fp);

    sample_count = 0;
    while (sample_count < MAX_SAMPLES) {
        double x, y;
        int n = fscanf(fp, "%lf,%lf", &x, &y);
        if (n != 2) break;  

        data[sample_count][0] = x;
        data[sample_count][1] = y;
        sample_count++;
    }

    fclose(fp);
    return 1;
}

/* ---------------- Normalize train x only ---------------- */
void normalize_train_x(size_t train_count)
{
    xmin = data[0][0];
    xmax = data[0][0];

    for (size_t i = 1; i < train_count; i++) {
        if (data[i][0] < xmin) xmin = data[i][0];
        if (data[i][0] > xmax) xmax = data[i][0];
    }

    double range = xmax - xmin;
    if (range == 0.0) range = 1.0;

    for (size_t i = 0; i < train_count; i++) {
        data_norm[i][0] = (data[i][0] - xmin) / range;
        data_norm[i][1] = data[i][1];
    }
}

/* ---------------- Gradient step ---------------- */
void gradient_step(double w, double b, size_t train_count,
                   double *dw, double *db, double *loss)
{
    *dw = *db = *loss = 0.0;

    for (size_t i = 0; i < train_count; i++) {
        double x = data_norm[i][0];
        double y = data_norm[i][1];
        double err = (w * x + b) - y;

        *loss += err * err;
        *dw += 2.0 * err * x;
        *db += 2.0 * err;
    }

    *loss /= train_count;
    *dw /= train_count;
    *db /= train_count;
}

int main(void)
{
    srand(8);

    if (!load_csv("linear-regression/data/sat-gpa.csv")) return 1;
    if (sample_count < 2) {
        printf("Not enough data.\n");
        return 1;
    }

    printf("Loaded %zu samples.\n", sample_count);

    /* split without shuffle */
    size_t train_count = (size_t)(sample_count * 0.75);
    size_t test_count = sample_count - train_count;
    printf("Training samples: %zu | Test samples: %zu\n", train_count, test_count);

    /* normalize train x */
    normalize_train_x(train_count);

    /* initialize model */
    double w = (double)rand() / RAND_MAX;
    double b = (double)rand() / RAND_MAX;
    double lr = 0.1;
    double last_loss = 1e30;

    /* train */
    for (size_t step = 0; step < 2000; step++) {
        double dw, db, loss;
        gradient_step(w, b, train_count, &dw, &db, &loss);

        if (loss > last_loss) lr *= 0.5;
        w -= lr * dw;
        b -= lr * db;

        if (step % 50 == 0)
            printf("step %zu | loss %.10f | w %.8f | b %.8f\n",
                   step, loss, w, b);

        if (fabs(last_loss - loss) < 1e-12) break;
        last_loss = loss;
    }

    /* convert to original x scale */
    double w_real = w / (xmax - xmin);
    double b_real = b - (w * xmin) / (xmax - xmin);
    printf("\nFinal model:\n");
    printf("GPA = %.6f * SAT + %.6f\n", w_real, b_real);

    /* ---------------- Evaluate on test set ---------------- */
    printf("\nTest set evaluation:\n");
    printf("SAT | Predicted GPA | Actual GPA | Abs Error | Error %%\n");
    printf("--------------------------------------------------------\n");

    for (size_t i = train_count; i < sample_count; i++) {
        double x_test = data[i][0];
        double y_true = data[i][1];
        double x_norm = (x_test - xmin) / (xmax - xmin);
        double y_pred = w * x_norm + b;

        double abs_err = fabs(y_pred - y_true);
        double rel_err = (fabs(y_true) > 1e-8)
            ? abs_err / fabs(y_true)
            : 0.0;

        printf("%4.0f | %13.3f | %10.3f | %9.3f | %7.2f%%\n",
               x_test, y_pred, y_true, abs_err, rel_err * 100.0);
    }

    printf("--------------------------------------------------------\n");
    printf("Enter a value for SAT to predict GPA: ");
    double sat_input;
    scanf("%lf", &sat_input);

    double x_norm = (sat_input - xmin) / (xmax - xmin);
    double y_pred = w * x_norm + b;

    printf("Predicted GPA for SAT = %.0f is %.3f\n", sat_input, y_pred);

    return 0;
}
