require "active_support/all"
require "torch-rb"

require_relative "activation"

class NeuralNetwork
  attr_reader :network

  def initialize
    @network = {}
    @network["W1"] = Torch.tensor([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    @network["b1"] = Torch.tensor([0.1, 0.2, 0.3])
    @network["W2"] = Torch.tensor([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    @network["b2"] = Torch.tensor([0.1, 0.2])
    @network["W3"] = Torch.tensor([[0.1, 0.3], [0.2, 0.4]])
    @network["b3"] = Torch.tensor([0.1, 0.2])
  end

  def forward(x)
    a1 = x.matmul(network["W1"]) + network["b1"]
    z1 = Activation.sigmoid(a1)
    a2 = z1.matmul(network["W2"]) + network["b2"]
    z2 = Activation.sigmoid(a2)
    a3 = z2.matmul(network["W3"]) + network["b3"]
    Activation.identity_function(a3)
  end
end
