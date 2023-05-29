`timescale 1ns / 1ps
// Code your testbench here
// or browse Examples
module D_FF_testbench;
reg CLK, reset, d;
wire q,q1;
parameter PERIOD = 1000;
  dffb m1(.q(q),.q1(q1),.d(d),.rst(reset),.clk(CLK)); // Instantiate the D_FF
initial CLK <= 0; // Set up clock
  always #1 CLK<= ~CLK;
initial begin // Set up signals
d = 0; 
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;
#1 d =1;
#5 d=0;
#2 d=1;  
#1000 $finish;  
end
  
  initial begin
    // Dump waves
    $dumpfile("dump.vcd");
    $dumpvars(1);
  end
endmodule
