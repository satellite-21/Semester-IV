`timescale 1ns / 1ps

module testBcdAdder;

    reg [3:0] a,b;
    reg carryInput;
    wire [3:0] sum;
    wire carry;
    bcdAdder bcd(a,b,carryInput,sum,carry);
    initial 
        begin
        
        a = 4'b0000;  b = 4'b0000;  carryInput = 1'b0; 
        
        #20 a = 4'b0011;  b = 4'b0010;
        #20 a = 4'b0100;  b = 4'b0011;
        #20 a = 4'b0110;  b = 4'b1000;
        #20 a = 4'b1011;  b = 4'b0010;
        #20 a = 4'b1000;  b = 4'b0101;
        end
        
    initial #100 $finish;
      
endmodule

