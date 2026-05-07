#import "CalculatorMath.h"

@implementation CalculatorMath
- (double)addValue:(double)a toValue:(double)b { return a + b; }
- (double)multiplyByFactor:(double)factor { return self.precision * factor; }
+ (instancetype)sharedInstance { static CalculatorMath *s; @synchronized(self) { if (!s) s = [[CalculatorMath alloc] init]; } return s; }
@end

@implementation CalculatorMath (Trig)
- (double)sineOfAngle:(double)angle { return sin(angle); }
- (double)cosineOfAngle:(double)angle { return cos(angle); }
@end
