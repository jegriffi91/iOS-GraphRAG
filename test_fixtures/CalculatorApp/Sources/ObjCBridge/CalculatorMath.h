#import <Foundation/Foundation.h>

@interface CalculatorMath : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) NSInteger precision;

- (double)addValue:(double)a toValue:(double)b;
- (double)multiplyByFactor:(double)factor;
+ (instancetype)sharedInstance;

@end

@interface CalculatorMath (Trig)
- (double)sineOfAngle:(double)angle;
- (double)cosineOfAngle:(double)angle;
@end

@protocol CalculatorMathProtocol <NSObject>
- (double)compute:(NSArray *)inputs;
@end
