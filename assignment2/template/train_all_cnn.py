"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""


def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))


def main(argv):
    pass


if __name__ == '__main__':
    main()
