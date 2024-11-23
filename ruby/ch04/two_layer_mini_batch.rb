require "matplotlib/pyplot"
require "torch-rb"
require_relative "../dataset/mnist"
require_relative "../common/utility"
require_relative "two_layer_net"

DEVICE = Torch::CUDA.available? ? "cuda" : "cpu" # can set device on new Torch.tensor

module TwoLayerMiniBatch
  module_function

  def perform # rubocop:disable Metrics/MethodLength
    x_train = Mnist.load_train_images(dtype: :float64, normalize: true)
    t_train = Mnist.load_train_labels
    # x_test = Mnist.load_test_images(normalize: true)
    # t_test = Mnist.load_test_labels

    train_loss_list = []

    # set train configs
    iters_num = 100 # 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet.new(input_size: 784, hidden_size: 50, output_size: 10)

    iters_num.times do |index|
      # Fetch mini-batch data
      batch_mask = Array(0..train_size.pred).sample(batch_size)
      x_batch = x_train[batch_mask]
      t_batch = t_train[batch_mask]

      # calculate gradient
      grad = network.numerical_gradient(x_batch, t_batch)
      # grad = network.gradient(x_batch, t_batch)

      # update params
      %w[W1 b1 W2 b2].each do |key|
        network.params[key] -= learning_rate * grad[key]
      end

      # save learning process data
      loss = network.loss(x_batch, t_batch)
      train_loss_list.append(loss)

      print index, "."
      $stdout.flush
    end
    print "done.\n"

    # draw learning process chart
    Utility.plot(Array(0..train_loss_list.length.pred), train_loss_list.map(&:to_f), ylim: nil)
  end
end

TwoLayerMiniBatch.perform
