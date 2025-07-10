# HB_GEN

🚧 Prototype system for HyperBlock-based classification model generation and early-stage evaluation across benchmark datasets.

## Project Goal

Achieve comparable performance with hyperblock models as to that of a Convolutional Neural Network on full real data i.e. MNIST.

### Datasets Used

- **Fisher Iris (4-D)**: Quick testing
  - Location: `datasets/fisher_iris.csv`
  - Samples: 150, Features: 4, Classes: 3

- **WBC (9-D and 30-D)**: Benchmark testing
  - Location: `datasets/wbc9.csv` and `datasets/wbc30.csv`
  - Samples: 571-684, Features: 9/30, Classes: 2

- **Dimensionally Reduced MNIST (121-D)**: Initial goal
  - Training: `datasets/mnist_dr_train.csv` (60,000 samples)
  - Test: `datasets/mnist_dr_test.csv` (10,000 samples)
  - Features: 121 (11×11 after cropping and 2×2 pooling)
  - Classes: 10 digits (0-9)

- **MNIST (784-D)**: Dimensionally reduced MNIST
  - Training: `datasets/mnist_dr_train.csv` (60,000 samples)
  - Test: `datasets/mnist_dr_test.csv` (10,000 samples)
  - Features: 784 (28×28 original pixels)
  - Classes: 10 digits (0-9)

- **MNIST by Class (784-D)**: Individual class datasets for smaller files, top-level goal
  - Training: `datasets/mnist_by_class/train/mnist_class_*_train.csv`
  - Test: `datasets/mnist_by_class/test/mnist_class_*_test.csv`
  - Features: 784 (28×28 original pixels)
  - Classes: 10 separate files (0-9)

## References

Algorithms used for HB generation from: [Visual Knowledge Discovery with General Line Coordinates](refs/VKD_with_GLC.pdf)

## License

This project is licensed for free commercial and personal use under the [MIT License](LICENSE).
