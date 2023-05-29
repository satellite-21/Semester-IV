`timescale 1ns / 1ps

module bcdAdder(input [3:0]a,input [3:0]b,input carryInput,output reg [3:0]sum,output reg carry);

    reg [4:0] sumTemp;
    
    always @(a,b,carryInput)
    begin
        sumTemp = a+b+carryInput; 
        if(sumTemp > 9)    
            begin
            sumTemp = sumTemp+6; 
            carry = 1;  
            sum = sumTemp[3:0];    
            end
        else    
            begin
            carry = 0;
            sum = sumTemp[3:0];
            end
    end

endmodule

