require "active_support/all"
require "torch-rb"
require "yaml"
require "zlib"

require_relative "activation"
require_relative "../dataset/mnist"

class NeuralNetworkMnist
  attr_reader :network

  def initialize
    @network = Zlib::GzipReader.open(File.expand_path("sample_weight.yml.gz", __dir__), &:read)
      .then { |text| YAML.safe_load(text) }
      .transform_values { |value| Torch.tensor(value) }
  end

  def predict(x)
    a1 = x.matmul(network["W1"]) + network["b1"]
    z1 = Activation.sigmoid(a1)
    a2 = z1.matmul(network["W2"]) + network["b2"]
    z2 = Activation.sigmoid(a2)
    a3 = z2.matmul(network["W3"]) + network["b3"]
    Activation.softmax(a3)
  end

  def calculate_accuracy
    images = Mnist.load_test_images(dtype: :float32)
    labels = Mnist.load_test_labels

    accuracy_cnt = 0
    images.each_with_index do |x, index|
      accuracy_cnt += 1 if Torch.equal(Torch.argmax(predict(x)), labels[index])
    end

    accuracy_cnt.to_f / labels.count
  end

  def calculate_batch_accuracy
    images = Mnist.load_test_images(dtype: :float32)
    labels = Mnist.load_test_labels

    batch_size = 100
    accuracy_cnt = 0
    (0...(images.size[0] / batch_size)).each do |index|
      range = (index * batch_size)...(index.succ * batch_size)
      x = images[range]
      accuracy_cnt += Torch.eq(Torch.argmax(predict(x), dim: 1), labels[range]).sum(&:long).to_i
    end

    accuracy_cnt.to_f / labels.count
  end
end
