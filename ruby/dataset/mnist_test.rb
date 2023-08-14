require_relative "../test/test_helper"
require_relative "mnist"

describe Mnist do
  it "should download files" do
    Mnist.load

    assert_path_exists File.expand_path("t10k-labels-idx1-ubyte.gz", __dir__)

    assert_equal \
      Mnist::FILE_CHECKSUMS["train-images-idx3-ubyte.gz"],
      Digest::MD5.file(File.expand_path("train-images-idx3-ubyte.gz", __dir__)).to_s
  end

  it "should shape 60000_784 for train images" do
    x_train = Mnist.load_train_images

    assert_equal [60000, 784], x_train.shape

    sample = x_train.first

    assert(sample.all? { |num| num.to_f.in?(0..255) })
  end

  it "should shape 60000 for train labels" do
    t_train = Mnist.load_train_labels

    assert_equal [60000], t_train.shape
  end

  it "should shape 10000_784 for test images" do
    x_test = Mnist.load_test_images

    assert_equal [10000, 784], x_test.shape
  end

  it "should shape 10000 for test labels" do
    t_test = Mnist.load_test_labels

    assert_equal [10000], t_test.shape
  end

  it "should support normalize on load image" do
    x_train = Mnist.load_train_images(normalize: true)
    sample = x_train.first

    assert(sample.all? { |num| num.to_f.in?(0..1) })
  end
end
