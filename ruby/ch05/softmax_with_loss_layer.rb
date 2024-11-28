require "torch-rb"
require_relative "../common/global"
require_relative "../ch04/loss"

class SoftmaxWithLossLayer
  attr_reader :loss, :y, :t

  def forward(x, t)
    @t = t
    @y = Torch.softmax(x, dim: 1)
    @loss = Loss.cross_entropy_error(y, t)

    loss
  end

  def backward(_dout = 1)
    batch_size = t.shape[0]
    (y - t) / batch_size
  end
end
