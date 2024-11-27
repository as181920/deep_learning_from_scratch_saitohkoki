require "torch-rb"
require_relative "../common/global"

class AffineLayer
  attr_reader :w, :b, :x, :dw, :db

  def initialize(w:, b:)
    @w = w
    @b = b
  end

  def forward(x)
    @x = x
    Torch::Linalg.multi_dot([x, w]) + b
  end

  def backward(dout)
    dx = Torch::Linalg.multi_dot([dout, w.transpose(0, 1)])
    @dw = Torch::Linalg.multi_dot([x.transpose(0, 1), dout])
    @db = Torch.sum(dout, 0)

    dx
  end
end
