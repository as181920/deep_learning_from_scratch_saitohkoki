require "matplotlib/pyplot"
require "torch-rb"
require_relative "../dataset/mnist"
require_relative "../common/global"
require_relative "../common/utility"
require_relative "two_layer_net"

DEVICE = Torch::CUDA.available? ? "cuda" : "cpu" # can set device on new Torch.tensor

module TwoLayerMiniBatch
  module_function

  def perform # rubocop:disable Metrics/MethodLength, Metrics/AbcSize
    x_train = Mnist.load_train_images(dtype: :float64, normalize: true).to(Global::DEVICE)
    t_train = Mnist.load_train_labels.to(Global::DEVICE)
    x_test = Mnist.load_test_images(dtype: :float64, normalize: true).to(Global::DEVICE)
    t_test = Mnist.load_test_labels.to(Global::DEVICE)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # set train configs
    iters_num = 100 # 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    iter_per_epoch = [(train_size / batch_size), 1].max

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

      # save train loss data
      loss = network.loss(x_batch, t_batch)
      train_loss_list.append(loss)

      # save acc data for each epoch
      if (index % iter_per_epoch).zero?
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print "\ntrain_acc: #{train_acc}, test_acc: #{test_acc}\n"
      end

      print index, "."
      $stdout.flush
    end
    print "done.\n"

    # draw train loss chart
    Utility.plot(Array(0..train_loss_list.length.pred), train_loss_list.map(&:to_f), ylim: nil, title: "Train Loss Data")

    # draw train acc chart
    # Utility.plot(Array(0..train_acc_list.length.pred), train_acc_list.map(&:to_f), ylim: nil) # data not enough when iters_num is small

    # draw test acc chart
    # Utility.plot(Array(0..test_acc_list.length.pred), test_acc_list.map(&:to_f), ylim: nil) # data not enough when iters_num is small
  end
end

TwoLayerMiniBatch.perform
