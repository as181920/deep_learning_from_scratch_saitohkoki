require_relative "../test/test_helper"
require_relative "loss"

describe Loss do
  it "should implement mean square error" do
    y = Torch.tensor([1.0, 2.0, 3.0])
    t = Torch.tensor([1.0, 2, 4])

    assert Torch.equal(Torch.tensor(0.5), Loss.mean_square_error(y, t))
  end

  it "should implement cross entropy error with one_hot" do
    y = Torch.tensor([1.0, 2.0, 3.0])
    t = Torch.tensor([1, 0, 0])

    assert_in_delta Torch.tensor(0.0).to_f, Loss.cross_entropy_error_with_one_hot(y, t).to_f, 10**-6
  end

  it "should implement cross entropy error with mini_batch one_hot" do
    y = Torch.tensor([[1, 2], [3, Torch.exp(Torch.tensor(1))]], dtype: :float)
    t = Torch.tensor([[1, 0], [0, 1]], dtype: :float)

    assert_in_delta (-Torch.tensor(1) / 2).to_f, Loss.cross_entropy_error_with_mini_batch_one_hot(y, t).to_f, 10**-6
  end

  it "should implement cross entropy error with mini_batch index" do
    y = Torch.tensor([[1, 2], [3, Torch.exp(Torch.tensor(1))]], dtype: :float)
    t = Torch.tensor([0, 1])

    assert_in_delta (-Torch.tensor(1) / 2).to_f, Loss.cross_entropy_error(y, t).to_f, 10**-6
  end
end
