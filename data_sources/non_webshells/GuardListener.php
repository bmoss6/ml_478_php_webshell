<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Workflow\EventListener;

use Symfony\Component\Security\Core\Authentication\AuthenticationTrustResolverInterface;
use Symfony\Component\Security\Core\Authentication\Token\Storage\TokenStorageInterface;
use Symfony\Component\Security\Core\Authorization\AuthorizationCheckerInterface;
use Symfony\Component\Security\Core\Role\Role;
use Symfony\Component\Security\Core\Role\RoleHierarchy;
use Symfony\Component\Security\Core\Role\RoleHierarchyInterface;
use Symfony\Component\Validator\Validator\ValidatorInterface;
use Symfony\Component\Workflow\Event\GuardEvent;
use Symfony\Component\Workflow\Exception\InvalidTokenConfigurationException;
use Symfony\Component\Workflow\TransitionBlocker;

/**
 * @author Grégoire Pineau <lyrixx@lyrixx.info>
 */
class GuardListener
{
    private $configuration;
    private $expressionLanguage;
    private $tokenStorage;
    private $authorizationChecker;
    private $trustResolver;
    private $roleHierarchy;
    private $validator;

    public function __construct(array $configuration, ExpressionLanguage $expressionLanguage, TokenStorageInterface $tokenStorage, AuthorizationCheckerInterface $authorizationChecker, AuthenticationTrustResolverInterface $trustResolver, RoleHierarchyInterface $roleHierarchy = null, ValidatorInterface $validator = null)
    {
        $this->configuration = $configuration;
        $this->expressionLanguage = $expressionLanguage;
        $this->tokenStorage = $tokenStorage;
        $this->authorizationChecker = $authorizationChecker;
        $this->trustResolver = $trustResolver;
        $this->roleHierarchy = $roleHierarchy;
        $this->validator = $validator;
    }

    public function onTransition(GuardEvent $event, $eventName)
    {
        if (!isset($this->configuration[$eventName])) {
            return;
        }

        $eventConfiguration = (array) $this->configuration[$eventName];
        foreach ($eventConfiguration as $guard) {
            if ($guard instanceof GuardExpression) {
                if ($guard->getTransition() !== $event->getTransition()) {
                    continue;
                }
                $this->validateGuardExpression($event, $guard->getExpression());
            } else {
                $this->validateGuardExpression($event, $guard);
            }
        }
    }

    private function validateGuardExpression(GuardEvent $event, string $expression)
    {
        if (!$this->expressionLanguage->evaluate($expression, $this->getVariables($event))) {
            $blocker = TransitionBlocker::createBlockedByExpressionGuardListener($expression);
            $event->addTransitionBlocker($blocker);
        }
    }

    // code should be sync with Symfony\Component\Security\Core\Authorization\Voter\ExpressionVoter
    private function getVariables(GuardEvent $event): array
    {
        $token = $this->tokenStorage->getToken();

        if (null === $token) {
            throw new InvalidTokenConfigurationException(sprintf('There are no tokens available for workflow %s.', $event->getWorkflowName()));
        }

        if (method_exists($token, 'getRoleNames')) {
            $roles = $token->getRoleNames();
        } else {
            $roles = array_map(function (Role $role) { return $role->getRole(); }, $token->getRoles(false));
        }

        if ($this->roleHierarchy instanceof RoleHierarchy) {
            $roles = $this->roleHierarchy->getReachableRoleNames($roles);
        } elseif (null !== $this->roleHierarchy) {
            $roles = $this->roleHierarchy->getReachableRoles($token->getRoles(false));
        }

        $variables = [
            'token' => $token,
            'user' => $token->getUser(),
            'subject' => $event->getSubject(),
            'roles' => $roles,
            // needed for the is_granted expression function
            'auth_checker' => $this->authorizationChecker,
            // needed for the is_* expression function
            'trust_resolver' => $this->trustResolver,
            // needed for the is_valid expression function
            'validator' => $this->validator,
        ];

        return $variables;
    }
}