`timescale 1ns / 1ps
`include "evm.v"
`include "display.v"

module test_bench_evm();
    reg A,B,enable,reset,display_result;
    wire [3:0]A_result,B_result;
    wire [0:6]A_display,B_display;
    
    evm e1(A,B,enable,reset,A_result,B_result);
    disp_result d1(A_result,B_result,A_display,B_display,display_result);
    
    initial
    begin
    reset=1;#10;
    reset=0;
    
    enable=1; A=1; B=0; display_result = 0; 
    #10 enable=0;
    
    #10 enable=1; A=1; B=0; display_result=0;
    #10; enable=0;
    
    #10; enable=0;
    display_result = 1;
    
    reset = 1;
    #10 reset = 0;
    
    enable=1; A=0; B=0; display_result=0; 
    #10; enable=0;
    #10 reset=0;enable=1; A=0; B=1;
    #10 enable=0; 
    #10 display_result=1;
    
    end
    initial #200 $finish;
    
endmodule
