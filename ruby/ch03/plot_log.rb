require_relative "../common/utility"

x = Torch.arange(0, 1, 0.001)
y = x.map { |e| Torch.log(e) }

Utility.plot(x.map(&:to_f), y.map(&:to_f), ylim: nil, title: "Log")
