`timescale 1ns / 1ps

module test_dff_blocking;

reg D, clk;
wire Q1, Q2;

dff_blocking db(D, clk, Q1, Q2);
initial
begin
    D = 0; clk = 0;
    #20 D = 1;
    #20 D = 0;
    #20 D = 1;
end


always #5 clk = ~clk;
initial #100 $finish;

endmodule
