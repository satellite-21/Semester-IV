`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
module dff_test_bench();
reg d1, d2, reset, clk;
wire o1, o2;

dff_async_sync_reset a1(d1, d2, reset, clk, o1, o2);

initial
begin

$monitor("Value of D1=%b, D2=%b, clk=%b, o1=%b, o2=%b", d1,d2, reset, clk, o1, o2);

clk = 0;
d1 = 0; d2 = 0; reset=0;

#2 d1 = 1; d2 = 1; reset = 0;
#2 d1 = 1; d2 = 1; reset = 1;
#2 d1 = 1; d2 = 1; reset = 0;

end
always #2 clk=~clk;
endmodule
