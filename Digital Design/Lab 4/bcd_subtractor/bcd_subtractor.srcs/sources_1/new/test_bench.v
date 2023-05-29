`timescale 1ns / 1ps
module test_bench;
wire [3:0] c;
wire s;
reg[3:0] a, b;
    
bcd_subtractor b1(a, b, c, s);
initial
        begin
        a = 7; b=7;
        #20 a = 3; b = 4;
        #20 a = 2; b = 7;
        #20 a = 3; b = 5; 
        end
initial #100 $finish;
endmodule   