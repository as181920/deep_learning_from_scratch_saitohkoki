require "torch-rb"
require_relative "../dataset/mnist"
require_relative "../common/global"
require_relative "../common/utility"
require_relative "two_layer_net"

module GradientCheck
  module_function

  def perform # rubocop:disable Metrics/MethodLength
    # load mnist data
    x_train = Mnist.load_train_images(dtype: :float64, normalize: true).to(Global::DEVICE)
    t_train = Mnist.load_train_labels(one_hot_label: true).to(Global::DEVICE)
    # x_test = Mnist.load_test_images(dtype: :float64, normalize: true).to(Global::DEVICE)
    # t_test = Mnist.load_test_labels(one_hot_label: true).to(Global::DEVICE)

    # set neuro network
    network = TwoLayerNet.new(input_size: 784, hidden_size: 50, output_size: 10)

    # get batch/test sample data
    x_batch = x_train[0..2]
    t_batch = t_train[0..2]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    grad_numerical.each_key do |grad_key|
      diff = (grad_backprop[grad_key] - grad_numerical[grad_key]).mean
      print grad_key, " diff: ", diff, "\n"
    end
  end
end

GradientCheck.perform
